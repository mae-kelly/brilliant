# 🚀 DeFi Momentum Trading System

## 📁 Repository Structure

```
├── core/                          # Core trading engine
│   ├── engine/                    # Main pipeline and orchestration
│   │   ├── pipeline.py           # Main trading pipeline
│   │   └── batch_processor.py    # High-performance token processing
│   ├── execution/                 # Trade execution and risk management
│   │   ├── trade_executor.py     # Trade execution logic
│   │   ├── risk_manager.py       # Risk management systems
│   │   └── scanner_v3.py         # Token scanning engine
│   ├── models/                    # ML models and inference
│   │   ├── inference_model.py    # Main ML inference engine
│   │   └── model_manager.py      # Model lifecycle management
│   └── features/                  # Feature engineering
│       └── vectorized_features.py # High-performance feature extraction
│
├── intelligence/                  # AI and analysis systems
│   ├── signals/                   # Signal detection
│   │   └── signal_detector.py    # Momentum signal detection
│   ├── analysis/                  # Advanced analysis
│   │   ├── advanced_ensemble.py  # Multi-modal analysis
│   │   ├── continuous_optimizer.py # Parameter optimization
│   │   └── feedback_loop.py      # Learning feedback loops
│   └── sentiment/                 # Sentiment analysis (if exists)
│
├── security/                      # Security and safety systems
│   ├── validators/                # Input validation and safety
│   │   ├── safety_checks.py      # Comprehensive safety validation
│   │   └── token_profiler.py     # Token analysis and profiling
│   ├── rugpull/                   # Rugpull detection
│   │   └── anti_rug_analyzer.py  # Advanced rugpull protection
│   └── mempool/                   # MEV and mempool protection
│       └── mempool_watcher.py    # Frontrunning and MEV protection
│
├── infrastructure/               # Infrastructure and ops
│   ├── config/                   # Configuration files
│   │   └── settings.yaml        # Main configuration
│   ├── monitoring/               # Monitoring and logging
│   │   ├── performance_optimizer.py # System optimization
│   │   ├── logging_config.py    # Logging configuration
│   │   └── error_handler.py     # Error handling utilities
│   └── deployment/               # Deployment configs
│       ├── docker-compose.yml   # Docker orchestration
│       └── docker/              # Docker configurations
│
├── data/                         # Data storage
│   ├── cache/                    # Database and cache files
│   └── models/                   # Trained model files
│
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Utility scripts
├── tests/                        # Test files
│
├── main.py                       # Main entry point
├── config.py                     # Unified configuration
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # Configure your keys
   ```

2. **Run Trading System:**
   ```bash
   python main.py
   ```

3. **Run in Notebook:**
   ```bash
   jupyter notebook notebooks/run_pipeline.ipynb
   ```

## 📊 Architecture

- **Core Engine**: High-performance trading pipeline
- **Intelligence**: AI-driven signal detection and analysis  
- **Security**: Comprehensive safety and rugpull protection
- **Infrastructure**: Monitoring, logging, and deployment

## 🔧 Configuration

All configuration is centralized in `infrastructure/config/settings.yaml` and can be overridden with environment variables in `.env`.
