# 🚀 DeFi Momentum Trading System - Production Guide

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- 8GB+ RAM
- 50GB+ disk space

### Deployment
```bash
# Clone and setup
git clone <repository>
cd defi-trading-system

# Run all optimization scripts
./1_core_fixes.sh
./2_intelligence_upgrade.sh
./3_production_deployment.sh

# Deploy to production
./deploy.sh production
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Token Scanner │    │  ML Inference   │    │ Trade Executor  │
│   Multi-Chain   │───▶│   TFLite        │───▶│  MEV Protected  │
│   Real-time     │    │   Ensemble      │    │   Smart Routes  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Safety Checks  │    │ Risk Management │    │  Feedback Loop  │
│  Rugpull/MEV    │    │  Portfolio      │    │ Online Learning │
│  Detection      │    │  VaR/Kelly      │    │ A/B Testing     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Performance Targets

| Metric | Target | Production SLA |
|--------|--------|----------------|
| Win Rate | >60% | >55% |
| Sharpe Ratio | >2.0 | >1.5 |
| Max Drawdown | <20% | <25% |
| Inference Latency | <100ms | <200ms |
| Trade Execution | <5s | <10s |
| Uptime | >99.5% | >99% |

## Monitoring

### Grafana Dashboards
- **System Health**: http://localhost:3000
- **Trading Performance**: Real-time P&L, win rate, Sharpe ratio
- **ML Model Metrics**: Prediction confidence, uncertainty, entropy
- **Risk Management**: VaR, portfolio exposure, gas costs

### Key Metrics to Monitor
1. **Trading Performance**
   - `trades_executed_total`: Trade volume
   - `trade_success_total`: Successful trades
   - `portfolio_exposure`: Current exposure by chain

2. **Model Performance**
   - `prediction_entropy`: Model uncertainty
   - `dynamic_threshold`: Adaptive threshold
   - `model_loss`: Training loss

3. **System Health**
   - `system_health`: Component health scores
   - `scan_latency_seconds`: Token scanning speed
   - `trade_latency_seconds`: Execution speed

## Configuration

### Environment Variables
```bash
# Blockchain RPCs
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_KEY

# Wallet
WALLET_ADDRESS=0x...
PRIVATE_KEY=0x...

# Trading
ENABLE_LIVE_TRADING=false  # Set to true for live trading
STARTING_BALANCE=0.01      # ETH
```

### Key Settings (settings.yaml)
```yaml
trading:
  base_position_size: 0.001
  momentum_threshold: 0.09
  velocity_threshold: 0.13
  max_slippage: 0.03

risk:
  max_position_size: 0.002
  max_portfolio_exposure: 0.01
  confidence_level: 0.95

ml:
  prediction_confidence: 0.75
  retrain_threshold: 500
```

## Safety Features

### Rugpull Protection
- LP lock verification
- Contract ownership analysis
- Pause function detection
- Holder distribution analysis

### MEV Protection
- Private mempool routing
- Frontrunning detection
- Sandwich attack prevention
- Gas price optimization

### Risk Management
- Dynamic position sizing (Kelly Criterion)
- Portfolio exposure limits
- Value at Risk monitoring
- Circuit breakers for extreme volatility

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats defi-trading
   
   # Restart if needed
   docker-compose restart defi-trading
   ```

2. **Model Inference Slow**
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Optimize model
   python model_manager.py --optimize
   ```

3. **Network Connection Issues**
   ```bash
   # Test RPC connections
   python validate_system.py
   
   # Check backup providers
   grep BACKUP .env
   ```

### Logs
```bash
# System logs
docker-compose logs -f defi-trading

# Specific component logs
docker-compose exec defi-trading tail -f /app/logs/trading.log
docker-compose exec defi-trading tail -f /app/logs/errors.log
```

## Optimization

### Model Optimization
The system automatically optimizes:
- **Dynamic Thresholds**: Based on recent performance
- **Ensemble Weights**: Multi-model optimization
- **Risk Parameters**: Kelly fraction, VaR calculations

### A/B Testing
- Automatic parameter testing
- Performance comparison
- Statistical significance testing

### Online Learning
- Continuous model retraining
- Performance feedback integration
- Adaptive threshold management

## Security

### Best Practices
1. **Never share private keys**
2. **Use environment variables for sensitive data**
3. **Regularly update dependencies**
4. **Monitor for unusual activity**
5. **Test with small amounts first**

### Wallet Security
- Use hardware wallet integration (future)
- Multi-signature support (future)
- Time-locked emergency procedures

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  defi-trading:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4GB
          cpus: '2'
```

### Performance Optimization
- GPU acceleration for ML inference
- Redis caching for data
- Connection pooling for RPCs
- Async processing for I/O

## API Reference

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"returns": 0.01, "volatility": 0.2, ...}'
```

### Metrics
```bash
curl http://localhost:8001/metrics
```

## Support

### Getting Help
1. Check logs first
2. Review monitoring dashboards
3. Validate configuration
4. Test individual components

### Performance Tuning
- Monitor Grafana dashboards
- Adjust configuration based on performance
- Use A/B testing for optimization
- Review and update ML models regularly

## License & Disclaimer

⚠️ **IMPORTANT**: This system is for educational and research purposes. 
Cryptocurrency trading involves significant risk. Never trade with funds you cannot afford to lose.
