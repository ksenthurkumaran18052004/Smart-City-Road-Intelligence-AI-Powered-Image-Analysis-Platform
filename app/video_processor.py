import cv2
import torch
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from app.models import (
    ModelLoader, AccidentDetector, HelmetDetector, 
    LicensePlateDetector, PotholeDetector, VehicleDetector
)
from app.ocr import LicensePlateOCR
from app.config import FRAME_SAMPLING_INTERVAL, ACCIDENT_THRESHOLD


class VideoAnalyzer:
    """
    Main video analysis pipeline
    Processes video frames and aggregates detection results
    """
    
    def __init__(self):
        """Initialize video analyzer with loaded models"""
        self.model_loader = ModelLoader()
        self.model_loader.load_all_models()
        
        # Initialize detector instances
        self.accident_detector = AccidentDetector(
            self.model_loader.get_model('accident')
        )
        self.helmet_detector = HelmetDetector(
            self.model_loader.get_model('helmet')
        )
        self.license_plate_detector = LicensePlateDetector(
            self.model_loader.get_model('license_plate')
        )
        self.pothole_detector = PotholeDetector(
            self.model_loader.get_model('pothole')
        )
        self.vehicle_detector = VehicleDetector(
            self.model_loader.get_model('vehicle')
        )
        
        # OCR module
        self.ocr = LicensePlateOCR()
    
    def analyze_video(self, video_path):
        """
        Complete video analysis pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with analysis results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Initialize results
        results = {
            'video_type': 'Normal',
            'accident_detected': False,
            'accident_confidence': 0.0,
            'helmet_violations': {
                'with_helmet': 0,
                'without_helmet': 0,
                'meaning': 'two_wheeler_observations'
            },
            'detected_plates': [],
            'potholes_detected': 0,
            'frames_analyzed': 0,
            'frame_events': [],
            'two_wheeler_events': [],
            'analysis_warnings': []
        }

        if self.accident_detector.model is None:
            results['analysis_warnings'].append(
                'Accident detection is unavailable: the accident checkpoint has no executable model graph.'
            )
        if self.pothole_detector.model is not None and self.pothole_detector.is_regression_model:
            results['analysis_warnings'].append(
                'Pothole detection is unavailable: the checkpoint predicts a box but has no objectness score.'
            )
        if self.vehicle_detector.model is None:
            results['analysis_warnings'].append(
                'Rider filtering is unavailable: no person/motorcycle detector is loaded.'
            )
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * FRAME_SAMPLING_INTERVAL) if fps > 0 else 1
        
        # Process frames
        frame_count = 0
        accident_predictions = []
        seen_plates = {}
        self._tracked_rider_bbox = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames at specified interval
            if frame_count % frame_interval == 0:
                # Preserve the original portrait layout so small plates are not
                # compressed by a forced 640x480 resize.
                processed_frame = frame
                
                # Run all detections
                self._process_frame(processed_frame, results, accident_predictions, seen_plates, frame_count / fps if fps else 0.0)
            
            frame_count += 1
        
        cap.release()

        # The same motorcycle appears in many sampled frames. Report unique
        # motorcycle tracks rather than counting every frame as a new rider.
        unique_riders = self._deduplicate_rider_events(results['two_wheeler_events'])
        results['two_wheeler_events'] = unique_riders
        results['helmet_violations']['with_helmet'] = sum(
            1 for event in unique_riders if event['helmet_status'] == 'with_helmet'
        )
        results['helmet_violations']['without_helmet'] = sum(
            1 for event in unique_riders if event['helmet_status'] == 'without_helmet'
        )
        
        # Aggregate accident predictions
        if accident_predictions:
            accident_ratio = sum(accident_predictions) / len(accident_predictions)
            results['accident_confidence'] = float(accident_ratio)
            results['accident_detected'] = accident_ratio >= ACCIDENT_THRESHOLD
            results['video_type'] = 'Accident' if results['accident_detected'] else 'Normal'
        
        # Deduplicate plates
        results['detected_plates'] = self._consolidate_plates(seen_plates.values())
        
        return results
    
    def _process_frame(self, frame, results, accident_predictions, seen_plates, timestamp):
        """
        Process a single frame with all detections
        
        Args:
            frame: OpenCV frame
            results: Results dictionary to update
            accident_predictions: List to append accident predictions
            seen_plates: Dictionary used to deduplicate recognized plates
        """
        with torch.no_grad():
            # 1. Accident Detection
            is_accident, confidence = self.accident_detector.predict(frame)
            accident_predictions.append(1 if is_accident else 0)
            
            # 2. Helmet Detection
            vehicle_detections = self.vehicle_detector.detect(frame)
            
            # 3. License Plate Detection & OCR
            plates = self.license_plate_detector.detect_plates(frame)
            frame_plates = []
            for plate_info in plates:
                plate_is_on_traffic = self._plate_belongs_to_two_wheeler(
                    plate_info['bbox'], vehicle_detections
                )
                if not plate_is_on_traffic and not self._plate_overlaps_person_body(
                    plate_info['bbox'], vehicle_detections
                ):
                    continue
                plate_region = plate_info['region']
                plate_conf = plate_info['confidence']
                
                # Run OCR
                plate_text, ocr_conf = self.ocr.recognize_plate(plate_region)
                
                if not plate_text and not (
                    plate_info['confidence'] >= 0.35
                    and plate_is_on_traffic
                ):
                    continue

                plate_entry = {
                    'plate_text': plate_text,
                    'confidence': min(float(plate_conf), float(ocr_conf)) if plate_text else float(plate_conf),
                    'detection_confidence': float(plate_conf),
                    'ocr_confidence': float(ocr_conf),
                    'bbox': list(plate_info['bbox']),
                    'status': 'recognized' if plate_text else 'detected_but_unreadable',
                }
                frame_plates.append(plate_entry)

                if plate_text:
                    if plate_text not in seen_plates:
                        seen_plates[plate_text] = plate_entry
                elif '__unreadable_plate__' not in seen_plates:
                    seen_plates['__unreadable_plate__'] = plate_entry

            helmet_detections = self._rider_helmet_detections(
                self.helmet_detector.detect(frame), vehicle_detections, frame_plates
            )
            
            # 4. Pothole Detection
            potholes = self.pothole_detector.detect_potholes(frame)
            results['potholes_detected'] += potholes
            rider_events = self._build_two_wheeler_events(vehicle_detections, helmet_detections, frame_plates)
            for event in rider_events:
                event['timestamp_seconds'] = round(float(timestamp), 2)
            results['two_wheeler_events'].extend(rider_events)
            results['frames_analyzed'] += 1
            results['frame_events'].append({
                'timestamp_seconds': round(float(timestamp), 2),
                'accident_detected': bool(is_accident),
                'accident_confidence': float(confidence),
                'helmets': helmet_detections,
                'plates': frame_plates,
                'riders': rider_events,
                'potholes_detected': int(potholes),
            })

    def _rider_helmet_detections(self, helmet_detections, vehicle_detections, plates=None):
        """Keep helmet boxes only when attached to a person on a two-wheeler."""
        people = [item for item in vehicle_detections if item['class_id'] == VehicleDetector.PERSON]
        two_wheelers = [
            item for item in vehicle_detections
            if item['class_id'] in (VehicleDetector.BICYCLE, VehicleDetector.MOTORCYCLE)
        ]
        plates = plates or []
        plate_people = []
        people_with_plate = []
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            for plate in plates:
                bx1, by1, bx2, by2 = plate['bbox']
                center_x = (bx1 + bx2) / 2
                center_y = (by1 + by2) / 2
                if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                    people_with_plate.append(person)
                    break

        selected_person = None
        if people_with_plate:
            selected_person = max(people_with_plate, key=lambda item: item['confidence'])
        elif self._tracked_rider_bbox is not None:
            selected_person = max(
                people,
                key=lambda item: self._bbox_iou(item['bbox'], self._tracked_rider_bbox),
                default=None,
            )
            if selected_person and self._bbox_iou(selected_person['bbox'], self._tracked_rider_bbox) < 0.1:
                selected_person = None

        rider_by_vehicle = {}
        candidate_people = [selected_person] if selected_person else people
        for person in candidate_people:
            if person is None:
                continue
            px1, py1, px2, py2 = person['bbox']
            person_width = px2 - px1
            person_height = py2 - py1
            person_center_x = (px1 + px2) / 2
            candidates = []
            for index, two_wheeler in enumerate(two_wheelers):
                tx1, ty1, tx2, ty2 = two_wheeler['bbox']
                horizontal_overlap = max(0, min(px2, tx2) - max(px1, tx1)) / max(1, min(person_width, tx2 - tx1))
                lower_body_overlap = max(0, min(py2, ty2) - max(py1 + person_height * 0.45, ty1))
                center_distance = abs(person_center_x - (tx1 + tx2) / 2) / max(1, tx2 - tx1)
                score = horizontal_overlap + lower_body_overlap / max(1, person_height) - center_distance * 0.25
                if horizontal_overlap >= 0.25 and lower_body_overlap > 0:
                    candidates.append((score, index))
            if candidates:
                _, vehicle_index = max(candidates)
                current = rider_by_vehicle.get(vehicle_index)
                if current is None or person['confidence'] > current['confidence']:
                    rider_by_vehicle[vehicle_index] = person

            if selected_person is person and not candidates and plates and two_wheelers:
                nearest_index = min(
                    range(len(two_wheelers)),
                    key=lambda index: abs(
                        ((two_wheelers[index]['bbox'][0] + two_wheelers[index]['bbox'][2]) / 2)
                        - ((px1 + px2) / 2)
                    )
                )
                rider_by_vehicle[nearest_index] = person

        if selected_person:
            self._tracked_rider_bbox = selected_person['bbox']

        associated_by_vehicle = {}
        for helmet in helmet_detections:
            hx1, hy1, hx2, hy2 = helmet['bbox']
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            for vehicle_index, person in rider_by_vehicle.items():
                px1, py1, px2, py2 = person['bbox']
                person_width = px2 - px1
                person_height = py2 - py1
                helmet_on_person = (
                    px1 <= hcx <= px2
                    and py1 <= hcy <= py1 + person_height * 0.45
                )
                if not helmet_on_person:
                    continue
                if vehicle_index in rider_by_vehicle:
                    two_wheeler = two_wheelers[vehicle_index]
                    helmet['rider_context'] = 'two_wheeler'
                    helmet['vehicle_bbox'] = two_wheeler['bbox']
                    current = associated_by_vehicle.get(vehicle_index)
                    if current is None or helmet['confidence'] > current['confidence']:
                        associated_by_vehicle[vehicle_index] = helmet
                    break
        return list(associated_by_vehicle.values())

    @staticmethod
    def _bbox_iou(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _consolidate_plates(entries):
        groups = []
        for entry in entries:
            text = entry.get('plate_text', '')
            if not text:
                continue
            matching_group = next(
                (group for group in groups if SequenceMatcher(None, text, group[0]['plate_text']).ratio() >= 0.78),
                None,
            )
            if matching_group is None:
                groups.append([entry])
            else:
                matching_group.append(entry)

        consolidated = []
        for group in groups:
            target_length = max(len(item.get('plate_text', '')) for item in group)
            consensus = ''.join(
                Counter(
                    item['plate_text'][index]
                    for item in group
                    if index < len(item.get('plate_text', ''))
                ).most_common(1)[0][0]
                for index in range(target_length)
            )
            best = max(
                group,
                key=lambda item: (
                    len(group),
                    len(item.get('plate_text', '')),
                    item.get('ocr_confidence', 0.0),
                    item.get('detection_confidence', 0.0),
                ),
            ).copy()
            best['plate_text'] = consensus
            best['observations'] = len(group)
            consolidated.append(best)
        return sorted(consolidated, key=lambda plate: plate['plate_text'])

    @staticmethod
    def _plate_belongs_to_two_wheeler(plate_bbox, vehicle_detections):
        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        for vehicle in vehicle_detections:
            if vehicle['class_id'] not in (
                VehicleDetector.BICYCLE,
                VehicleDetector.CAR,
                VehicleDetector.MOTORCYCLE,
                VehicleDetector.BUS,
                VehicleDetector.TRUCK,
            ):
                continue
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            vehicle_width = vx2 - vx1
            vehicle_height = vy2 - vy1
            if (
                vx1 - vehicle_width * 0.15 <= plate_center[0] <= vx2 + vehicle_width * 0.15
                and vy1 - vehicle_height * 0.15 <= plate_center[1] <= vy2 + vehicle_height * 0.15
            ):
                return True
        return False

    @staticmethod
    def _plate_overlaps_person_body(plate_bbox, vehicle_detections):
        """Allow a strong plate box when COCO missed the motorcycle itself."""
        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        for person in vehicle_detections:
            if person['class_id'] != VehicleDetector.PERSON:
                continue
            x1, y1, x2, y2 = person['bbox']
            if x1 <= plate_center[0] <= x2 and y1 + (y2 - y1) * 0.45 <= plate_center[1] <= y2:
                return True
        return False

    @staticmethod
    def _build_two_wheeler_events(vehicle_detections, helmet_detections, plates):
        events = []
        two_wheelers = [
            item for item in vehicle_detections
            if item['class_id'] in (VehicleDetector.BICYCLE, VehicleDetector.MOTORCYCLE)
        ]
        for two_wheeler in two_wheelers:
            vx1, vy1, vx2, vy2 = two_wheeler['bbox']
            helmets = [
                helmet for helmet in helmet_detections
                if helmet.get('vehicle_bbox') == two_wheeler['bbox']
            ]
            matched_plates = []
            for plate in plates:
                px1, py1, px2, py2 = plate['bbox']
                center = ((px1 + px2) / 2, (py1 + py2) / 2)
                if vx1 <= center[0] <= vx2 and vy1 <= center[1] <= vy2:
                    matched_plates.append(
                        plate['plate_text'] or 'plate detected; text unreadable'
                    )
            if helmets:
                events.append({
                    'vehicle_type': 'motorcycle' if two_wheeler['class_id'] == VehicleDetector.MOTORCYCLE else 'bicycle',
                    'helmet_status': helmets[0]['label'],
                    'plate_numbers': matched_plates,
                    'vehicle_bbox': two_wheeler['bbox'],
                })
        unreadable_plate = next(
            (plate for plate in plates if plate['status'] == 'detected_but_unreadable'),
            None,
        )
        if unreadable_plate and len(events) == 1 and not events[0]['plate_numbers']:
            events[0]['plate_numbers'].append('plate detected; text unreadable')
        return events

    @staticmethod
    def _deduplicate_rider_events(events):
        unique = []
        for event in events:
            x1, y1, x2, y2 = event['vehicle_bbox']
            matched = None
            for existing in unique:
                ex1, ey1, ex2, ey2 = existing['vehicle_bbox']
                intersection = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
                union = (x2 - x1) * (y2 - y1) + (ex2 - ex1) * (ey2 - ey1) - intersection
                if union and intersection / union >= 0.5:
                    matched = existing
                    break
            if matched is None:
                unique.append(event)
            else:
                if event['helmet_status'] == 'without_helmet':
                    matched['helmet_status'] = 'without_helmet'
                for plate in event['plate_numbers']:
                    if plate not in matched['plate_numbers']:
                        matched['plate_numbers'].append(plate)
        return unique


def validate_video_file(file_path):
    """
    Validate video file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    from app.config import ALLOWED_VIDEO_EXTENSIONS
    
    file_path = Path(file_path)
    
    # Check extension
    if file_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return False
    
    # Check if file exists and is readable
    if not file_path.exists() or not file_path.is_file():
        return False
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(str(file_path))
    is_valid = cap.isOpened()
    cap.release()
    
    return is_valid
