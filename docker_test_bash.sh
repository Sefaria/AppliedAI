#!/bin/bash

# Define image name
IMAGE_NAME="your-app-image"

# Build the image
docker build -t $IMAGE_NAME .

# Check if Docker build succeeded
if [ $? -eq 0 ]; then
    echo "Docker build succeeded."
else
    echo "Docker build failed."
    exit 1
fi
