import os
import sys
import time
from production_config import production_config
from monitoring import monitor
from risk_manager import risk_manager

def validate_production_environment():
    print("=== PRODUCTION DEPLOYMENT VALIDATION ===")
    
    ready, issues = production_config.validate_production_ready()
    
    if not ready:
        print("❌ Production validation FAILED:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    try:
        from secure_loader import config
        config.validate_all()
        print("✅ Configuration validated")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    required_env_vars = [
        'MAX_POSITION_USD',
        'MAX_DAILY_LOSS_USD',
        'DISCORD_WEBHOOK_URL'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ All production checks passed")
    return True

def deploy_to_production():
    if not validate_production_environment():
        print("Deployment aborted due to validation failures")
        return False
    
    print("\n=== STARTING PRODUCTION DEPLOYMENT ===")
    
    confirmation = input("Type 'DEPLOY_TO_PRODUCTION' to confirm: ")
    if confirmation != 'DEPLOY_TO_PRODUCTION':
        print("Deployment cancelled")
        return False
    
    print("Starting monitoring systems...")
    monitor.start_monitoring()
    
    print("Initializing risk management...")
    risk_manager.emergency_risk_check()
    
    print("✅ Production deployment successful!")
    print(f"📊 Monitoring dashboard: http://localhost:8090")
    print(f"💰 Max position size: ${production_config.limits.max_position_usd}")
    print(f"🛑 Max daily loss: ${production_config.limits.max_daily_loss_usd}")
    
    return True

if __name__ == "__main__":
    success = deploy_to_production()
    sys.exit(0 if success else 1)
