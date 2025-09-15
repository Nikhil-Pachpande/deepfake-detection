# Deployment Guide

This guide will help you deploy the Deepfake Detection application to various hosting platforms.

## Prerequisites

- Node.js 16+ installed
- Python 3.9+ installed
- Git repository (GitHub, GitLab, or Bitbucket)

## Step 1: Prepare Your Repository

1. **Push your code to a Git repository**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Ensure all files are included**
   - `backend/` directory with Python files
   - `frontend/` directory with React app
   - `build.sh` script

## Step 2: Choose Your Deployment Platform

### Option A: Docker Deployment

1. **Create a Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       libopencv-dev \
       python3-opencv \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY backend/requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY backend/ .
   
   # Expose port
   EXPOSE 8000
   
   # Run the application
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run**
   ```bash
   docker build -t deepfake-detector .
   docker run -p 8000:8000 deepfake-detector
   ```

### Option B: Heroku Deployment

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Ubuntu/Debian
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

3. **Create Procfile**
   ```
   web: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

### Option C: AWS EC2 Deployment

1. **Launch EC2 instance**
   - Choose Ubuntu 20.04 LTS
   - Configure security groups (open ports 22, 80, 443, 8000)

2. **Connect and setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python 3.9
   sudo apt install python3.9 python3.9-pip python3.9-venv -y
   
   # Install Node.js
   curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Install OpenCV dependencies
   sudo apt install libopencv-dev python3-opencv -y
   ```

3. **Deploy application**
   ```bash
   # Clone repository
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection
   
   # Setup backend
   cd backend
   python3.9 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Run with Gunicorn
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
   ```

### Option D: Google Cloud Platform

1. **Create App Engine application**
   ```yaml
   # app.yaml
   runtime: python39
   
   env_variables:
     PORT: 8000
   
   handlers:
   - url: /.*
     script: auto
   ```

2. **Deploy**
   ```bash
   gcloud app deploy
   ```

## Step 3: Environment Variables

Create environment variables for production:

```env
# Production settings
MAX_FILE_SIZE=52428800  # 50MB
CORS_ORIGINS=https://yourdomain.com
DEBUG=False
```

## Step 4: Frontend Deployment

### Option A: Static Hosting (Netlify, Vercel, GitHub Pages)

1. **Build the frontend**
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy build folder**
   - Upload `build/` folder to your static hosting service
   - Configure API endpoint to point to your backend

### Option B: Serve with Backend

1. **Update backend to serve static files**
   ```python
   from fastapi.staticfiles import StaticFiles
   
   app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
   ```

## Step 5: Domain and SSL

1. **Configure DNS**
   - Point your domain to your server's IP address
   - Set up A records or CNAME records

2. **SSL Certificate**
   - Use Let's Encrypt for free SSL certificates
   - Configure your web server (Nginx/Apache) for HTTPS

## Troubleshooting

### Common Issues

**Build Fails**
- Check Python version compatibility
- Verify all dependencies are in `requirements.txt`
- Ensure OpenCV dependencies are installed

**API Not Working**
- Check firewall settings
- Verify port configuration
- Review application logs

**Frontend Not Loading**
- Ensure build directory is created
- Check API endpoint configuration
- Verify CORS settings

### Logs and Debugging

1. **View application logs**
   ```bash
   # For systemd services
   journalctl -u your-service-name -f
   
   # For Docker
   docker logs container-name
   
   # For direct execution
   tail -f app.log
   ```

2. **Check system resources**
   ```bash
   # CPU and memory usage
   htop
   
   # Disk space
   df -h
   
   # Network connections
   netstat -tulpn
   ```

## Performance Optimization

### Backend Optimization

1. **Use production WSGI server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```

2. **Enable caching**
   - Use Redis for session storage
   - Implement response caching

3. **Database optimization**
   - Use connection pooling
   - Optimize database queries

### Frontend Optimization

1. **Enable gzip compression**
2. **Optimize images and assets**
3. **Use CDN for static assets**
4. **Implement lazy loading**

## Security

1. **Environment Variables**: Never commit sensitive data
2. **CORS Configuration**: Restrict origins in production
3. **Rate Limiting**: Implement for high traffic
4. **Input Validation**: Ensure all inputs are validated
5. **HTTPS**: Always use SSL in production
6. **Firewall**: Configure proper firewall rules

## Monitoring

1. **Application Monitoring**: Use tools like New Relic, DataDog
2. **Log Aggregation**: ELK stack or similar
3. **Uptime Monitoring**: Pingdom, UptimeRobot
4. **Error Tracking**: Sentry, Rollbar

## Backup and Recovery

1. **Code Repository**: Keep your code in version control
2. **Database Backups**: Regular automated backups
3. **Configuration Backups**: Document all environment variables
4. **Disaster Recovery Plan**: Test your recovery procedures

## Cost Estimation

### Cloud Platform Costs (Monthly)

- **AWS EC2 t3.micro**: ~$8-10
- **Google Cloud f1-micro**: ~$5-7
- **Heroku Hobby**: $7
- **DigitalOcean Droplet**: $5-10

### Additional Costs

- **Domain**: $10-15/year
- **SSL Certificate**: Free (Let's Encrypt)
- **Monitoring**: $0-50/month (depending on service)

## Support

- **Documentation**: Check platform-specific documentation
- **Community Forums**: Stack Overflow, Reddit
- **Professional Support**: Consider managed hosting services

---

**Happy Deploying! 🚀**