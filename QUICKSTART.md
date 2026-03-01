# Smart City AI Platform - Quick Start Guide

## Installation

1. **Navigate to project**:
   ```bash
   cd smart_city_ai
   ```

2. **Create virtual environment**:
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
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki (default install path works)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

## Running the Application

```bash
python -m app.main
```

Open browser: **http://localhost:8000**

## Features

✅ Accident Detection
✅ Helmet Violation Detection  
✅ License Plate Recognition
✅ Pothole Detection
✅ Clean Web Dashboard
✅ Responsive Design

## Requirements Met

✅ Backend API (FastAPI)
✅ Frontend UI (HTML/CSS/JS)
✅ Video Analysis Pipeline
✅ Modular Architecture
✅ Production-Ready Code
✅ Proper Error Handling
✅ GPU Support
✅ All Models Integrated
