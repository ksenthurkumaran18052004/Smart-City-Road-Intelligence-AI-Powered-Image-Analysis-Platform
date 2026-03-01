# Smart City Road Intelligence Platform

Complete AI-powered backend and frontend for automatic video analysis with multiple detection models.

## Features

- **🚨 Accident Detection**: Classifies videos as accident or normal based on frame analysis
- **🪖 Helmet Violation Detection**: Detects riders with/without helmets
- **🚔 License Plate Recognition**: Detects and recognizes license plates using YOLO + OCR
- **🕳️ Pothole Detection**: Counts pothole detections in video
- **📊 Clean Dashboard**: Modern responsive web UI for results display
- **⚡ Production-Ready**: Modular, scalable architecture with proper error handling

## Project Structure

```
smart_city_ai/
├── app/
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py            # Configuration settings
│   ├── models.py            # Model loading & inference
│   ├── video_processor.py    # Video analysis pipeline
│   └── ocr.py              # License plate OCR
├── models/                  # Pre-trained model files (.pt)
├── static/
│   └── styles.css          # Frontend styling
├── templates/
│   ├── index.html          # Upload page
│   └── results.html        # Results display
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- Tesseract OCR engine installed

### Setup Steps

1. **Clone/Navigate to project directory**:
   ```bash
   cd smart_city_ai
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**:
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Mac**: `brew install tesseract`
   
   Update path in `app/ocr.py` if needed:
   ```python
   pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

5. **Verify model files** are in `models/` directory:
   - `accident_detection_model.pt`
   - `helmet_detection_model.pt`
   - `license_plate_recognization_model.pt`
   - `pothole_detection_model.pt`

## Running the Application

### Start the server:

```bash
python -m app.main
```

The server will start on `http://localhost:8000`

### Access the web interface:

- **Upload page**: `http://localhost:8000/`
- **API health check**: `http://localhost:8000/api/health`

## Usage

1. **Visit the web interface** at `http://localhost:8000/`
2. **Upload a video** by dragging and dropping or clicking to select
3. **Wait for analysis** (progress indicator shown)
4. **View results** in the dashboard with:
   - Accident status and confidence
   - Helmet violation statistics
   - Detected license plates with OCR text
   - Pothole count

## API Endpoints

### GET `/`
Renders the upload page

### POST `/upload-video`
Upload and analyze video

**Request**: Form data with video file
```bash
curl -X POST -F "file=@video.mp4" http://localhost:8000/upload-video
```

**Response**:
```json
{
  "video_type": "Accident",
  "accident_confidence": 0.91,
  "helmet_violations": {
    "with_helmet": 12,
    "without_helmet": 3
  },
  "detected_plates": [
    {
      "plate_text": "TN09AB1234",
      "confidence": 0.88
    }
  ],
  "potholes_detected": 5,
  "filename": "video.mp4",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### GET `/results`
Renders the results page

### GET `/api/health`
Health check endpoint

```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda"
}
```

## Configuration

Edit `app/config.py` to customize:

- `FRAME_SAMPLING_INTERVAL`: Process 1 frame per X seconds (default: 0.5)
- `ACCIDENT_THRESHOLD`: Accident classification threshold (default: 0.7)
- `MAX_UPLOAD_SIZE`: Maximum file size in bytes (default: 500MB)
- `ALLOWED_VIDEO_EXTENSIONS`: Supported video formats
- `DEVICE`: Force CPU/GPU selection

## Video Processing Details

### Frame Sampling
- Extracts 1 frame every 0.5 seconds (configurable)
- Does NOT process every frame for efficiency
- Processes frames at 640x480 resolution

### Accident Detection
- Runs classification on sampled frames
- If >70% (configurable) frames predict accident → classified as "Accident"
- Returns confidence score (0-1)

### Helmet Detection
- Detects riders in each frame
- Counts with helmet vs. without helmet
- Aggregates counts across all frames

### License Plate Recognition
- YOLO object detection on license plates
- Crops detected regions
- OCR using Tesseract
- Cleans text with regex filtering
- Returns unique plates with confidence

### Pothole Detection
- YOLO object detection
- Counts total detections across all frames

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is installed for faster inference
2. **Video Duration**: Shorter videos (~1 min) process in 30-60 seconds
3. **Model Loading**: Models load once on startup - subsequent videos process faster
4. **Memory**: GPU memory ~4GB, CPU ~2GB

## Troubleshooting

### Models not loading
- Verify `.pt` files exist in `models/` directory
- Check file permissions
- Ensure PyTorch installation matches your CUDA version

### Tesseract not found
- Verify installation path in `app/ocr.py`
- On Windows, add to PATH or set explicit path

### Out of memory
- Reduce frame size in `video_processor.py`: 
  ```python
  processed_frame = cv2.resize(frame, (480, 360))
  ```
- Increase `FRAME_SAMPLING_INTERVAL` for fewer frames

### GPU not detected
- Install CUDA 11.8+
- Reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## Deployment

For production deployment:

1. Use a production ASGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
   ```

2. Configure environment variables (create `.env`):
   ```
   DEVICE=cuda
   MAX_UPLOAD_SIZE=500000000
   ```

3. Use a reverse proxy (nginx/Apache)

4. Add SSL/TLS certificates

## License

Proprietary - Smart City Patent

## Support

For issues or questions, review logs in console output.

---

**Built with**: FastAPI, PyTorch, YOLO, OpenCV, Tesseract
