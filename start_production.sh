#!/bin/bash

echo "Starting DeFi Trading Bot in PRODUCTION mode..."

if [ ! -f ".env.production" ]; then
    echo "❌ .env.production file not found"
    echo "Copy .env.production template and configure with real values"
    exit 1
fi

export $(cat .env.production | xargs)

python3 deploy_production.py

if [ $? -eq 0 ]; then
    echo "🚀 Starting production services..."
    
    python3 health_check.py &
    HEALTH_PID=$!
    
    python3 start_bot.py
    
    kill $HEALTH_PID 2>/dev/null
else
    echo "❌ Production deployment failed"
    exit 1
fi
