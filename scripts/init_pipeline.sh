#!/bin/bash

echo "🏗️ Initializing Renaissance Pipeline..."

python -m pip install --upgrade pip
pip install -r requirements_final.txt

mkdir -p {logs,cache,models,data,charts,backup}

if [ ! -f "models/model_weights.tflite" ]; then
    echo "🧠 Training initial model..."
    python synthetic_training_data.py
fi

if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cat > .env << ENVEOF
ALCHEMY_API_KEY=your_alchemy_api_key_here
WALLET_ADDRESS=0x0000000000000000000000000000000000000000
PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000000
DRY_RUN=true
ENABLE_REAL_TRADING=false
MAX_POSITION_USD=10.0
ENVEOF
fi

chmod +x *.sh

echo "✅ Pipeline initialized successfully!"
echo "🚀 Run: python run_production_system.py --duration 0.5"
