#!/bin/bash

# Production deployment script for DeFi Trading System
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DEPLOYMENT_ENV=${1:-production}
DOCKER_REGISTRY=${2:-"defi-trading"}

echo -e "${BLUE}🚀 Deploying DeFi Trading System to ${DEPLOYMENT_ENV}${NC}"

# Pre-deployment checks
echo -e "${YELLOW}🔍 Running pre-deployment checks...${NC}"

# Check required files
required_files=(".env" "settings.yaml" "requirements.txt")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}❌ Missing required file: $file${NC}"
        exit 1
    fi
done

# Validate configuration
echo -e "${YELLOW}🔧 Validating configuration...${NC}"
python3 validate_system.py || {
    echo -e "${RED}❌ Configuration validation failed${NC}"
    exit 1
}

# Run tests
echo -e "${YELLOW}🧪 Running test suite...${NC}"
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short || {
        echo -e "${RED}❌ Tests failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  pytest not available, skipping tests${NC}"
fi

# Build Docker image
echo -e "${YELLOW}🐳 Building Docker image...${NC}"
docker build -t ${DOCKER_REGISTRY}:${DEPLOYMENT_ENV} . || {
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
}

# Security scan (if available)
if command -v docker-scout &> /dev/null; then
    echo -e "${YELLOW}🔒 Running security scan...${NC}"
    docker scout cves ${DOCKER_REGISTRY}:${DEPLOYMENT_ENV} || {
        echo -e "${YELLOW}⚠️  Security vulnerabilities detected${NC}"
    }
fi

# Deploy with docker-compose
echo -e "${YELLOW}🚢 Deploying services...${NC}"
docker-compose down --remove-orphans || true
docker-compose up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Trading API is healthy${NC}"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts - waiting for API..."
    sleep 10
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}❌ Services failed to become healthy${NC}"
    docker-compose logs
    exit 1
fi

# Post-deployment verification
echo -e "${YELLOW}🔍 Running post-deployment verification...${NC}"

# Test API endpoints
api_tests=(
    "http://localhost:8000/health"
    "http://localhost:8001/metrics"
    "http://localhost:9090/-/healthy"  # Prometheus
    "http://localhost:3000/api/health" # Grafana
)

for endpoint in "${api_tests[@]}"; do
    if curl -sf "$endpoint" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $endpoint${NC}"
    else
        echo -e "${RED}❌ $endpoint${NC}"
    fi
done

# Test model inference
test_payload='{"returns": 0.01, "volatility": 0.2, "momentum": 0.05, "rsi": 60, "bb_position": 0.5, "volume_ma": 5000, "whale_activity": 0.1, "price_acceleration": 0.001, "volatility_ratio": 1.0, "momentum_strength": 0.08, "swap_volume": 5000}'

if curl -sf -X POST http://localhost:8000/predict \
   -H "Content-Type: application/json" \
   -d "$test_payload" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Model inference test passed${NC}"
else
    echo -e "${RED}❌ Model inference test failed${NC}"
fi

echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}📊 Access points:${NC}"
echo "  🔗 Trading API: http://localhost:8000"
echo "  📈 Metrics: http://localhost:8001/metrics"
echo "  📊 Prometheus: http://localhost:9090"
echo "  📋 Grafana: http://localhost:3000 (admin/admin)"
echo "  💻 System Monitor: http://localhost:8080"

echo -e "\n${BLUE}📋 Next steps:${NC}"
echo "  1. Monitor system health in Grafana dashboard"
echo "  2. Fund wallet with initial capital (0.01+ ETH)"
echo "  3. Enable live trading: set ENABLE_LIVE_TRADING=true"
echo "  4. Monitor logs: docker-compose logs -f defi-trading"
