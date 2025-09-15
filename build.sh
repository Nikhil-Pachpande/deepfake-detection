#!/bin/bash

echo "Building Deepfake Detector..."

# Build frontend
echo "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

# Copy frontend build to root
echo "Copying frontend build files..."
cp -r frontend/build .

echo "Build complete! Ready for deployment."
echo ""
echo "The application can now be deployed to any hosting platform."
echo "See DEPLOYMENT.md for detailed deployment instructions."
