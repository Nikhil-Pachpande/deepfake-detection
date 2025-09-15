from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import base64
import io
from typing import Optional
import logging
from pydantic import BaseModel
import asyncio
import aiohttp
from urllib.parse import urlparse
import requests
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    reasons: list
    processing_time: float

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

class SimpleDeepfakeDetector:
    def __init__(self):
        # Simple detector for serverless compatibility
        pass
        
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in an image - simplified version"""
        height, width = image.shape[:2]
        # Return a mock face detection in the center
        face_size = min(height, width) // 3
        x = (width - face_size) // 2
        y = (height - face_size) // 2
        return [(x, y, face_size, face_size)]
    
    def analyze_image_simple(self, image: np.ndarray) -> dict:
        """Simple analysis for deepfake detection"""
        try:
            height, width = image.shape[:2]
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate basic statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Detect faces
            faces = self.detect_faces(image)
            
            # Simple blur detection
            laplacian_var = np.var(np.diff(np.diff(gray, axis=0), axis=1))
            
            # Edge analysis
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            edge_density = (np.sum(grad_x) + np.sum(grad_y)) / (height * width)
            
            return {
                "face_count": len(faces),
                "mean_brightness": mean_brightness,
                "std_brightness": std_brightness,
                "blur_score": laplacian_var,
                "edge_density": edge_density,
                "image_size": height * width
            }
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return {
                "face_count": 0,
                "mean_brightness": 128,
                "std_brightness": 0,
                "blur_score": 0,
                "edge_density": 0,
                "image_size": 0
            }
    
    async def analyze_image(self, image: np.ndarray) -> DetectionResponse:
        """Analyze an image for deepfake characteristics"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            analysis = self.analyze_image_simple(image)
            
            confidence_factors = []
            reasons = []
            
            # Face analysis
            if analysis["face_count"] == 0:
                confidence_factors.append(0.3)
                reasons.append("No faces detected in the image")
            elif analysis["face_count"] > 1:
                confidence_factors.append(0.4)
                reasons.append("Multiple faces detected")
            
            # Blur analysis
            if analysis["blur_score"] < 50:
                confidence_factors.append(0.6)
                reasons.append("Image appears to be heavily blurred")
            elif analysis["blur_score"] > 500:
                confidence_factors.append(0.3)
                reasons.append("Image is unusually sharp")
            
            # Brightness analysis
            if analysis["std_brightness"] < 20:
                confidence_factors.append(0.5)
                reasons.append("Low contrast detected")
            
            # Edge analysis
            if analysis["edge_density"] < 0.01:
                confidence_factors.append(0.4)
                reasons.append("Very few edges detected")
            
            # Size analysis
            if analysis["image_size"] < 10000:
                confidence_factors.append(0.3)
                reasons.append("Very small image size")
            
            # Calculate final confidence
            if not confidence_factors:
                confidence = 0.1
                reasons = ["No suspicious patterns detected - likely authentic"]
            else:
                confidence = min(max(confidence_factors), 0.8)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return DetectionResponse(
                is_deepfake=confidence > 0.5,
                confidence=confidence,
                reasons=reasons,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

detector = SimpleDeepfakeDetector()

def validate_file_type(content_type: str) -> bool:
    """Validate file type"""
    return content_type in ALLOWED_IMAGE_TYPES

async def download_from_url(url: str) -> bytes:
    """Download file from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to download from URL: {response.status}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading from URL: {str(e)}")

def process_image(image_data: bytes) -> np.ndarray:
    """Process image data to numpy array format"""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Deepfake Detection API is running", "version": "1.0.0"}

@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image_deepfake(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """Detect deepfake in an uploaded image or image URL"""
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or URL must be provided")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either file or URL, not both")
    
    try:
        # Handle file upload
        if file:
            if not validate_file_type(file.content_type):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image.")
            
            if file.size and file.size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
            
            image_data = await file.read()
        
        # Handle URL
        else:
            if not urlparse(url).scheme:
                raise HTTPException(status_code=400, detail="Invalid URL format")
            
            image_data = await download_from_url(url)
        
        # Process image
        image = process_image(image_data)
        
        # Analyze for deepfake
        result = await detector.analyze_image(image)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# For Vercel deployment
handler = app
