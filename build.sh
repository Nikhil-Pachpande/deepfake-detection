#!/bin/bash

echo "Building Deepfake Detector for Vercel deployment..."

# Build frontend
echo "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

# Copy frontend build to root for Vercel
echo "Copying frontend build files..."
cp -r frontend/build .

echo "Build complete! Ready for Vercel deployment."
echo ""
echo "To deploy:"
echo "1. Install Vercel CLI: npm i -g vercel"
echo "2. Run: vercel"
echo "3. Follow the prompts"
