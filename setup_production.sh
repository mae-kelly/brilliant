#!/bin/bash

echo "🚀 Setting up Renaissance DeFi Trading System"
echo "=============================================="

pip install -r requirements_final.txt

mkdir -p logs cache models data/backup

echo "✅ Setup complete! Run the system with:"
echo "   python run_production_system.py --duration 0.5"
echo "   python run_production_system.py --duration 24 --target 15000"
