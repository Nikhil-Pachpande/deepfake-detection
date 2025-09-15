# Deepfake Detection Web Application

A modern web application that uses AI-powered computer vision to detect potential deepfake content in images and videos. Built with FastAPI (Python) backend and React frontend, designed for easy deployment on Vercel.

![Deepfake Detection App](https://via.placeholder.com/800x400/2563eb/ffffff?text=Deepfake+Detection+Tool)

## 🚀 Features

- **Multi-format Support**: Analyze images (JPEG, PNG, WebP) and videos (MP4, AVI, MOV)
- **Multiple Input Methods**: Upload files directly or provide URLs
- **Real-time Analysis**: Fast AI-powered deepfake detection
- **Detailed Results**: Confidence scores and detailed reasoning
- **Modern UI**: Clean, responsive interface with drag-and-drop functionality
- **Vercel Ready**: Optimized for serverless deployment

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing
- **aiohttp**: Asynchronous HTTP client/server

### Frontend
- **React 18**: Modern React with hooks
- **React Dropzone**: File upload with drag-and-drop
- **Custom CSS**: Tailwind-inspired styling
- **Axios**: HTTP client for API calls

### Deployment
- **Vercel**: Serverless deployment platform
- **Python 3.9**: Backend runtime
- **Node.js**: Frontend build environment

## 📋 Prerequisites

- Node.js 16+ and npm
- Python 3.9+
- Git

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deepfake-detector
   ```

2. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```
   The backend will be available at `http://localhost:8000`

3. **Set up the frontend** (in a new terminal)
   ```bash
   cd frontend
   npm install
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

### Production Deployment on Vercel

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Build and deploy**
   ```bash
   # From the project root
   chmod +x build.sh
   ./build.sh
   vercel
   ```

3. **Follow Vercel prompts**
   - Link to existing project or create new one
   - Configure environment variables if needed
   - Deploy!

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Optional: API keys for external services
# OPENAI_API_KEY=your_openai_key
# CUSTOM_MODEL_URL=your_model_endpoint

# Optional: Custom settings
MAX_FILE_SIZE=52428800  # 50MB in bytes
CORS_ORIGINS=*  # For production, specify your domain
```

### API Endpoints

#### Health Check
```http
GET /
```
Returns API status and version information.

#### Image Detection
```http
POST /detect/image
Content-Type: multipart/form-data

file: [image file] OR url: "https://example.com/image.jpg"
```

#### Video Detection
```http
POST /detect/video
Content-Type: multipart/form-data

file: [video file] OR url: "https://example.com/video.mp4"
```

### Response Format

```json
{
  "is_deepfake": false,
  "confidence": 0.15,
  "reasons": [
    "No suspicious patterns detected - likely authentic"
  ],
  "processing_time": 0.234
}
```

## 🧠 How It Works

### Detection Methods

The application uses multiple computer vision techniques to detect potential deepfakes:

1. **Face Detection**: Uses OpenCV's Haar cascades to detect faces
2. **Blur Analysis**: Laplacian variance to measure image sharpness
3. **Edge Analysis**: Canny edge detection for artifact identification
4. **Frequency Analysis**: FFT-based analysis for compression artifacts
5. **Color Consistency**: LAB color space analysis for inconsistencies
6. **Lighting Analysis**: Brightness distribution analysis

### Confidence Scoring

- **0.0 - 0.3**: Likely authentic
- **0.3 - 0.7**: Uncertain, requires manual review
- **0.7 - 1.0**: High probability of being manipulated

## 📁 Project Structure

```
deepfake-detector/
├── backend/
│   ├── main.py              # Main FastAPI application
│   ├── vercel_main.py       # Vercel-optimized version
│   ├── requirements.txt     # Python dependencies
│   └── vercel.json         # Vercel configuration
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── App.js          # Main application
│   │   └── App.css         # Styling
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
├── build.sh               # Build script
├── vercel.json           # Root Vercel configuration
└── README.md             # This file
```

## 🔒 Security Considerations

- **File Size Limits**: Maximum 50MB per upload
- **File Type Validation**: Only allowed image/video formats
- **CORS Configuration**: Configurable cross-origin policies
- **Input Sanitization**: URL validation and file type checking
- **Rate Limiting**: Consider implementing for production use

## 🚨 Limitations & Disclaimers

### Technical Limitations

- **Accuracy**: No deepfake detection system is 100% accurate
- **False Positives/Negatives**: May occur due to various factors
- **Model Limitations**: Uses basic computer vision techniques
- **Video Analysis**: Simplified frame-by-frame analysis

### Legal Disclaimer

This tool is provided for educational and research purposes only. Users should:

- Not rely solely on automated detection results
- Verify important information through multiple sources
- Consult with experts for critical decisions
- Understand that results may not be admissible in legal proceedings

## 🔮 Future Enhancements

- **Advanced AI Models**: Integration with state-of-the-art deepfake detection models
- **Real-time Processing**: WebRTC-based real-time video analysis
- **Batch Processing**: Multiple file upload and analysis
- **API Authentication**: Secure API access with authentication
- **Model Training**: Custom model training capabilities
- **Mobile App**: React Native mobile application
- **Browser Extension**: Chrome/Firefox extension for quick analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [API Docs](https://your-vercel-app.vercel.app/docs)
- **Email**: support@yourdomain.com

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- FastAPI team for the excellent web framework
- React team for the frontend framework
- Vercel for serverless deployment platform

---

**Built with ❤️ for media authenticity and digital trust**
