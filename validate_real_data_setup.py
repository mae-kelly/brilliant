import os
import asyncio
from web3 import Web3

async def validate_setup():
    print("🔍 Validating Real Data Setup...")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ['ALCHEMY_API_KEY', 'PRIVATE_KEY', 'WALLET_ADDRESS']
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your_') or value in ['demo_key', '0x0000000000000000000000000000000000000000']:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing/invalid environment variables: {missing_vars}")
        print("Please set these properly:")
        for var in missing_vars:
            print(f"export {var}=your_actual_value")
        return False
    else:
        print("✅ Environment variables configured")
    
    # Test Web3 connections
    alchemy_key = os.getenv('ALCHEMY_API_KEY')
    
    chains = {
        'ethereum': f'https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}',
        'arbitrum': f'https://arb-mainnet.g.alchemy.com/v2/{alchemy_key}',
        'polygon': f'https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}'
    }
    
    for chain, rpc in chains.items():
        try:
            w3 = Web3(Web3.HTTPProvider(rpc))
            if w3.is_connected():
                latest_block = w3.eth.block_number
                print(f"✅ {chain}: Connected (Block: {latest_block})")
            else:
                print(f"❌ {chain}: Connection failed")
                return False
        except Exception as e:
            print(f"❌ {chain}: Error - {e}")
            return False
    
    # Test wallet balance
    try:
        wallet_address = os.getenv('WALLET_ADDRESS')
        w3 = Web3(Web3.HTTPProvider(chains['ethereum']))
        balance = w3.eth.get_balance(wallet_address)
        balance_eth = w3.from_wei(balance, 'ether')
        print(f"💰 Wallet balance: {balance_eth:.6f} ETH")
        
        if float(balance_eth) < 0.005:
            print("⚠️  Warning: Very low ETH balance")
        else:
            print("✅ Sufficient ETH balance")
            
    except Exception as e:
        print(f"❌ Wallet validation failed: {e}")
        return False
    
    print("=" * 50)
    print("🎉 Real data setup validation complete!")
    return True

if __name__ == "__main__":
    result = asyncio.run(validate_setup())
    exit(0 if result else 1)
