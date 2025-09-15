from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import requests
from PIL import Image
import cv2
import numpy as np
import base64
import io
from typing import Optional, Union
import logging
from pydantic import BaseModel
import asyncio
import aiohttp
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionRequest(BaseModel):
    url: Optional[str] = None
    media_type: str  # "image" or "video"

class DetectionResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    reasons: list
    processing_time: float

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mov", "video/quicktime"]

class DeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def analyze_face_consistency(self, image: np.ndarray, faces: list) -> dict:
        """Analyze face consistency for deepfake detection"""
        analysis = {
            "face_count": len(faces),
            "face_size_variation": 0,
            "blur_analysis": 0,
            "lighting_inconsistency": 0
        }
        
        if len(faces) > 1:
            # Check face size variation
            face_areas = [w * h for x, y, w, h in faces]
            analysis["face_size_variation"] = np.std(face_areas) / np.mean(face_areas) if face_areas else 0
        
        # Analyze first face for blur and lighting
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            # Blur analysis using Laplacian variance
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            analysis["blur_analysis"] = blur_score
            
            # Lighting analysis
            face_mean = np.mean(gray_face)
            analysis["lighting_inconsistency"] = abs(face_mean - 128) / 128
        
        return analysis
    
    def detect_artifacts(self, image: np.ndarray) -> dict:
        """Detect common deepfake artifacts"""
        artifacts = {
            "color_inconsistency": 0,
            "edge_anomalies": 0,
            "frequency_analysis": 0
        }
        
        # Color inconsistency analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        artifacts["color_inconsistency"] = np.std(l_channel) / np.mean(l_channel)
        
        # Edge analysis
        edges = cv2.Canny(image, 50, 150)
        artifacts["edge_anomalies"] = np.sum(edges) / (image.shape[0] * image.shape[1])
        
        # Frequency domain analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        artifacts["frequency_analysis"] = np.std(magnitude_spectrum)
        
        return artifacts
    
    async def analyze_image(self, image: np.ndarray) -> DetectionResponse:
        """Analyze an image for deepfake characteristics"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Detect faces
            faces = self.detect_faces(image)
            
            # Analyze face consistency
            face_analysis = self.analyze_face_consistency(image, faces)
            
            # Detect artifacts
            artifact_analysis = self.detect_artifacts(image)
            
            # Calculate overall confidence
            confidence_factors = []
            reasons = []
            
            # Face-based analysis
            if face_analysis["face_count"] == 0:
                confidence_factors.append(0.3)
                reasons.append("No faces detected in the image")
            elif face_analysis["face_count"] > 1:
                if face_analysis["face_size_variation"] > 0.3:
                    confidence_factors.append(0.7)
                    reasons.append("Inconsistent face sizes detected")
            
            # Blur analysis
            if face_analysis["blur_analysis"] < 100:
                confidence_factors.append(0.6)
                reasons.append("Unusually low sharpness detected")
            
            # Lighting analysis
            if face_analysis["lighting_inconsistency"] > 0.3:
                confidence_factors.append(0.5)
                reasons.append("Inconsistent lighting patterns")
            
            # Artifact analysis
            if artifact_analysis["color_inconsistency"] > 0.2:
                confidence_factors.append(0.6)
                reasons.append("Color inconsistencies detected")
            
            if artifact_analysis["edge_anomalies"] > 0.1:
                confidence_factors.append(0.7)
                reasons.append("Edge artifacts detected")
            
            if artifact_analysis["frequency_analysis"] > 5:
                confidence_factors.append(0.8)
                reasons.append("Frequency domain anomalies")
            
            # Calculate final confidence
            if not confidence_factors:
                confidence = 0.1
                reasons = ["No suspicious patterns detected - likely authentic"]
            else:
                confidence = max(confidence_factors)
            
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

detector = DeepfakeDetector()

def validate_file_type(content_type: str, media_type: str) -> bool:
    """Validate file type"""
    if media_type == "image":
        return content_type in ALLOWED_IMAGE_TYPES
    elif media_type == "video":
        return content_type in ALLOWED_VIDEO_TYPES
    return False

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
    """Process image data to OpenCV format"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and then to OpenCV format
        image_array = np.array(image)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
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
            if not validate_file_type(file.content_type, "image"):
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

@app.post("/detect/video", response_model=DetectionResponse)
async def detect_video_deepfake(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """Detect deepfake in an uploaded video or video URL"""
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or URL must be provided")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either file or URL, not both")
    
    try:
        # Handle file upload
        if file:
            if not validate_file_type(file.content_type, "video"):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid video.")
            
            if file.size and file.size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
            
            video_data = await file.read()
        
        # Handle URL
        else:
            if not urlparse(url).scheme:
                raise HTTPException(status_code=400, detail="Invalid URL format")
            
            video_data = await download_from_url(url)
        
        # For video analysis, we'll extract frames and analyze them
        # This is a simplified implementation - in production, you'd want more sophisticated video analysis
        
        # Save temporary video file
        temp_video_path = "/tmp/temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_data)
        
        # Extract frames and analyze
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count == 0:
            raise HTTPException(status_code=400, detail="Could not read video file")
        
        # Analyze key frames (every 30th frame or at least 5 frames)
        frame_interval = max(1, frame_count // 5)
        analyzed_frames = []
        
        for frame_idx in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_result = await detector.analyze_image(frame)
                analyzed_frames.append(frame_result)
        
        cap.release()
        
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Aggregate results from all frames
        if not analyzed_frames:
            raise HTTPException(status_code=400, detail="No valid frames found in video")
        
        # Calculate average confidence and combine reasons
        avg_confidence = sum(frame.confidence for frame in analyzed_frames) / len(analyzed_frames)
        all_reasons = []
        for frame in analyzed_frames:
            all_reasons.extend(frame.reasons)
        
        # Remove duplicates while preserving order
        unique_reasons = list(dict.fromkeys(all_reasons))
        
        # Add video-specific analysis
        if frame_count > 1000:
            unique_reasons.append("Long video duration - higher likelihood of manipulation")
        
        processing_time = sum(frame.processing_time for frame in analyzed_frames)
        
        return DetectionResponse(
            is_deepfake=avg_confidence > 0.5,
            confidence=avg_confidence,
            reasons=unique_reasons[:10],  # Limit to 10 most relevant reasons
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
