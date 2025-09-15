# Deployment Guide

This guide will help you deploy the Deepfake Detection application to Vercel.

## Prerequisites

- Node.js 16+ installed
- Python 3.9+ installed
- Vercel account (free tier available)
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
   - `vercel.json` configuration file
   - `build.sh` script

## Step 2: Install Vercel CLI

```bash
npm i -g vercel
```

## Step 3: Deploy to Vercel

### Option A: Deploy via Vercel CLI

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy the project**
   ```bash
   vercel
   ```

3. **Follow the prompts**
   - Link to existing project or create new one
   - Choose your Git provider
   - Configure settings:
     - **Framework Preset**: Other
     - **Root Directory**: `./`
     - **Build Command**: `./build.sh`
     - **Output Directory**: `build`

### Option B: Deploy via Vercel Dashboard

1. **Go to [vercel.com](https://vercel.com)**
2. **Click "New Project"**
3. **Import your Git repository**
4. **Configure settings**:
   - Framework: Other
   - Root Directory: `./`
   - Build Command: `chmod +x build.sh && ./build.sh`
   - Output Directory: `build`

## Step 4: Environment Variables (Optional)

If you need custom configuration, add environment variables in Vercel dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add variables:
   - `MAX_FILE_SIZE`: `52428800` (50MB)
   - `CORS_ORIGINS`: `*` (or your domain)

## Step 5: Custom Domain (Optional)

1. Go to your project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Configure DNS records as instructed

## Troubleshooting

### Common Issues

**Build Fails**
- Check that `build.sh` has execute permissions
- Verify all dependencies are in `requirements.txt`
- Check Python version compatibility

**API Not Working**
- Verify Vercel configuration in `vercel.json`
- Check function timeout settings
- Review logs in Vercel dashboard

**Frontend Not Loading**
- Ensure `build` directory is created
- Check that React build completed successfully
- Verify API endpoints are correctly configured

### Logs and Debugging

1. **View deployment logs**
   ```bash
   vercel logs
   ```

2. **Check function logs in Vercel dashboard**
   - Go to Functions tab
   - Click on individual functions
   - View real-time logs

## Performance Optimization

### Backend Optimization

1. **Use `vercel_main.py`** for production (optimized for serverless)
2. **Reduce package size** by using `opencv-python-headless`
3. **Set appropriate timeouts** in `vercel.json`

### Frontend Optimization

1. **Enable gzip compression**
2. **Optimize images** and assets
3. **Use CDN** for static assets

## Scaling Considerations

### Vercel Limits (Free Tier)
- 100GB bandwidth per month
- 100GB-hours of function execution
- 12 serverless functions

### Upgrading to Pro
- Higher bandwidth limits
- More function execution time
- Priority support
- Custom domains

## Monitoring

1. **Vercel Analytics**: Built-in performance monitoring
2. **Function Metrics**: Monitor API response times
3. **Error Tracking**: Set up error monitoring service

## Security

1. **Environment Variables**: Never commit sensitive data
2. **CORS Configuration**: Restrict origins in production
3. **Rate Limiting**: Consider implementing for high traffic
4. **Input Validation**: Ensure all inputs are validated

## Backup and Recovery

1. **Git Repository**: Keep your code in version control
2. **Database**: If using external databases, ensure backups
3. **Environment Variables**: Document all environment variables

## Cost Estimation

### Vercel Free Tier
- Perfect for development and small projects
- No cost for personal use
- Limited bandwidth and function execution

### Vercel Pro ($20/month)
- Suitable for production applications
- Higher limits and better performance
- Custom domains and advanced features

## Support

- **Vercel Documentation**: [vercel.com/docs](https://vercel.com/docs)
- **Community Forum**: [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)
- **Support Email**: [support@vercel.com](mailto:support@vercel.com)

---

**Happy Deploying! 🚀**
