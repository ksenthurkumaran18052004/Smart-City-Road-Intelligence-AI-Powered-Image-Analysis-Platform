from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import json
from datetime import datetime

from app.config import UPLOAD_DIR, ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE, API_TITLE, API_VERSION
from app.image_processor import ImageAnalyzer, validate_image_file

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-powered Smart City Road Intelligence Platform"
)

# Mount static files
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize image analyzer (loads models once at startup)
image_analyzer = ImageAnalyzer()

# Template directory
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    """
    Render the image upload page
    """
    try:
        with open(TEMPLATE_DIR / "index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Upload page not found")


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and analyze image
    
    Args:
        file: Image file from user
        
    Returns:
        JSON response with analysis results
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )
    
    # Create temporary file
    temp_file_path = UPLOAD_DIR / f"temp_{datetime.now().timestamp()}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            
            # Check file size
            if len(content) > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {MAX_UPLOAD_SIZE / (1024*1024):.0f} MB"
                )
            
            buffer.write(content)
        
        # Validate image file
        if not validate_image_file(temp_file_path):
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        # Analyze image
        print(f"Analyzing image: {file.filename}")
        results = image_analyzer.analyze_image(temp_file_path)
        results['filename'] = file.filename
        results['timestamp'] = datetime.now().isoformat()
        
        print(f"Analysis complete for: {file.filename}")
        print(f"Results: {json.dumps(results, indent=2)}")
        
        return JSONResponse(content=results)
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except:
                pass


@app.get("/results", response_class=HTMLResponse)
async def get_results_page():
    """
    Render the results display page
    """
    try:
        with open(TEMPLATE_DIR / "results.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Results page not found")


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "models_loaded": True,
        "device": "cuda" if image_analyzer.model_loader._loaded else "cpu"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print(f"{API_TITLE} v{API_VERSION}")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("Upload endpoint: POST /upload-image")
    print("UI endpoint: GET /")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
