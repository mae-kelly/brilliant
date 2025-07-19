import sys
import os
import time
from dev_mode import dev_wrapper

def safety_check():
    print("=== TRADING BOT SAFETY CHECK ===")
    
    if not os.path.exists('.env'):
        print("ERROR: .env file not found")
        return False
    
    try:
        from secure_loader import config
        if not config.validate_all():
            print("ERROR: Configuration validation failed")
            return False
    except Exception as e:
        print(f"ERROR: Configuration error: {e}")
        return False
    
    if dev_wrapper.config.enable_real_trading:
        print("WARNING: Real trading is ENABLED")
        confirm = input("Type 'CONFIRM_LIVE_TRADING' to proceed: ")
        if confirm != 'CONFIRM_LIVE_TRADING':
            print("Live trading not confirmed. Exiting.")
            return False
    else:
        print("SAFE: Running in development/simulation mode")
    
    return True

def simulate_pipeline():
    print("Starting trading bot simulation...")
    print("- Initializing scanner...")
    time.sleep(1)
    print("- Loading ML model...")
    time.sleep(1)
    print("- Starting token monitoring...")
    
    try:
        for i in range(5):
            print(f"Cycle {i+1}: Scanning for tokens...")
            time.sleep(2)
            if dev_wrapper.config.dry_run:
                print("  [DRY RUN] Found potential token, simulating trade...")
                dev_wrapper.simulate_trade(f"TOKEN_{i+1}", 1.0, "buy")
                time.sleep(1)
                dev_wrapper.simulate_trade(f"TOKEN_{i+1}", 1.05, "sell")
            print(f"  Balance: ${dev_wrapper.simulated_balance:.2f}")
        
        print("\nSimulation complete!")
        stats = dev_wrapper.get_stats()
        print(f"Final balance: ${stats['simulated_balance']:.2f}")
        print(f"Total trades: {stats['total_trades']}")
        
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if not safety_check():
        sys.exit(1)
    
    print("Starting trading bot...")
    simulate_pipeline()
