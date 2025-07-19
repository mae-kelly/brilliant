#!/bin/bash

echo "Cleaning up development files..."

rm -rf __pycache__
rm -rf .pytest_cache
rm -rf *.pyc
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Cleaned up Python cache files"

if [ "$1" = "full" ]; then
    echo "Full cleanup - removing logs and cache..."
    rm -rf logs/*
    rm -rf cache/*
    rm -rf models/*.h5
    rm -rf models/*.tflite
    echo "Full cleanup complete"
fi
