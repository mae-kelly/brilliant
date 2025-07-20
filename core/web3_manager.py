import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from web3 import Web3
from web3.exceptions import Web3Exception, ContractLogicError
from eth_account import Account
from eth_utils import to_checksum_address, is_address
import os
from decimal import Decimal
import math

@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int
    total_supply: int
    owner: str
    is_paused: bool
    has_mint_function: bool
    has_burn_function: bool
    max_tx_amount: int
    liquidity_locked: bool

@dataclass
class PairInfo:
    address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    total_supply: int
    factory: str

class Web3Manager:
    def __init__(self):
        self.providers = {}
        self.accounts = {}
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        self.chains = {
            'ethereum': {
                'rpc': f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
                'chain_id': 1,
                'weth': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'usdc': '0xA0b86a33E6441545C1F45DAB67F5d1C52bcfC8f4',
                'factories': {
                    'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                    'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                    'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'
                }
            },
            'arbitrum': {
                'rpc': f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
                'chain_id': 42161,
                'weth': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                'usdc': '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',
                'factories': {
                    'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                    'camelot': '0x6EcCab422D763aC031210895C81787E87B91425a'
                }
            },
            'polygon': {
                'rpc': f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
                'chain_id': 137,
                'weth': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
                'usdc': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
                'factories': {
                    'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                    'quickswap': '0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32'
                }
            }
        }
        
        self.erc20_abi = [
            {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "owner", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "paused", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
            {"constant": False, "inputs": [{"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "transfer", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
            {"constant": False, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"}
        ]
        
        self.pair_abi = [
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "_reserve0", "type": "uint112"}, {"name": "_reserve1", "type": "uint112"}, {"name": "_blockTimestampLast", "type": "uint32"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}
        ]
        
        self.factory_abi = [
            {"constant": True, "inputs": [{"name": "tokenA", "type": "address"}, {"name": "tokenB", "type": "address"}], "name": "getPair", "outputs": [{"name": "pair", "type": "address"}], "type": "function"}
        ]

    async def initialize(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        for chain, config in self.chains.items():
            try:
                self.providers[chain] = Web3(Web3.HTTPProvider(config['rpc']))
                if self.providers[chain].is_connected():
                    block = self.providers[chain].eth.block_number
                    self.logger.info(f"Connected to {chain} - Block: {block}")
                    
                    private_key = os.getenv('PRIVATE_KEY')
                    if private_key and len(private_key) == 66:
                        self.accounts[chain] = Account.from_key(private_key)
                        balance = self.providers[chain].eth.get_balance(self.accounts[chain].address)
                        self.logger.info(f"Account {self.accounts[chain].address} - Balance: {Web3.from_wei(balance, 'ether'):.4f} ETH")
                else:
                    self.logger.error(f"Failed to connect to {chain}")
            except Exception as e:
                self.logger.error(f"Error connecting to {chain}: {e}")

    async def get_token_info(self, token_address: str, chain: str) -> Optional[TokenInfo]:
        if not is_address(token_address):
            return None
            
        w3 = self.providers.get(chain)
        if not w3:
            return None
        
        try:
            token_address = to_checksum_address(token_address)
            contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            calls = [
                contract.functions.symbol(),
                contract.functions.name(),
                contract.functions.decimals(),
                contract.functions.totalSupply()
            ]
            
            try:
                symbol = calls[0].call()
                name = calls[1].call()
                decimals = calls[2].call()
                total_supply = calls[3].call()
            except:
                return None
            
            owner = '0x0000000000000000000000000000000000000000'
            is_paused = False
            has_mint = False
            has_burn = False
            max_tx = total_supply
            
            try:
                owner = contract.functions.owner().call()
            except:
                pass
                
            try:
                is_paused = contract.functions.paused().call()
            except:
                pass
            
            bytecode = w3.eth.get_code(token_address)
            bytecode_hex = bytecode.hex()
            
            has_mint = 'mint' in bytecode_hex.lower() or '40c10f19' in bytecode_hex
            has_burn = 'burn' in bytecode_hex.lower() or '42966c68' in bytecode_hex
            
            return TokenInfo(
                address=token_address,
                symbol=symbol,
                name=name,
                decimals=decimals,
                total_supply=total_supply,
                owner=owner,
                is_paused=is_paused,
                has_mint_function=has_mint,
                has_burn_function=has_burn,
                max_tx_amount=max_tx,
                liquidity_locked=False
            )
            
        except Exception as e:
            self.logger.error(f"Error getting token info for {token_address}: {e}")
            return None

    async def get_pair_info(self, token0: str, token1: str, chain: str, factory_name: str = 'uniswap_v2') -> Optional[PairInfo]:
        w3 = self.providers.get(chain)
        if not w3:
            return None
            
        factory_address = self.chains[chain]['factories'].get(factory_name)
        if not factory_address:
            return None
            
        try:
            factory = w3.eth.contract(address=to_checksum_address(factory_address), abi=self.factory_abi)
            pair_address = factory.functions.getPair(to_checksum_address(token0), to_checksum_address(token1)).call()
            
            if pair_address == '0x0000000000000000000000000000000000000000':
                return None
                
            pair_contract = w3.eth.contract(address=pair_address, abi=self.pair_abi)
            
            reserves = pair_contract.functions.getReserves().call()
            actual_token0 = pair_contract.functions.token0().call()
            actual_token1 = pair_contract.functions.token1().call()
            total_supply = pair_contract.functions.totalSupply().call()
            
            return PairInfo(
                address=pair_address,
                token0=actual_token0,
                token1=actual_token1,
                reserve0=reserves[0],
                reserve1=reserves[1],
                total_supply=total_supply,
                factory=factory_address
            )
            
        except Exception as e:
            self.logger.error(f"Error getting pair info: {e}")
            return None

    async def get_token_price(self, token_address: str, chain: str) -> Optional[float]:
        weth_address = self.chains[chain]['weth']
        
        pair = await self.get_pair_info(token_address, weth_address, chain)
        if not pair:
            return None
            
        token_info = await self.get_token_info(token_address, chain)
        if not token_info:
            return None
            
        if pair.token0.lower() == token_address.lower():
            token_reserve = pair.reserve0
            weth_reserve = pair.reserve1
        else:
            token_reserve = pair.reserve1
            weth_reserve = pair.reserve0
            
        if token_reserve == 0:
            return None
            
        weth_per_token = (weth_reserve * (10 ** token_info.decimals)) / (token_reserve * (10 ** 18))
        
        eth_price = await self.get_eth_price()
        if not eth_price:
            eth_price = 2500
            
        return weth_per_token * eth_price

    async def get_eth_price(self) -> Optional[float]:
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['ethereum']['usd']
        except Exception as e:
            self.logger.error(f"Error getting ETH price: {e}")
        return None

    async def analyze_contract_security(self, token_address: str, chain: str) -> Dict[str, Any]:
        w3 = self.providers.get(chain)
        if not w3:
            return {'safe': False, 'reasons': ['No provider']}
            
        try:
            bytecode = w3.eth.get_code(to_checksum_address(token_address))
            bytecode_hex = bytecode.hex().lower()
            
            risks = []
            
            dangerous_functions = [
                ('setTaxFeePercent', 'can change tax'),
                ('excludeFromReward', 'can exclude from rewards'),
                ('blacklistAddress', 'can blacklist addresses'),
                ('setMaxTxPercent', 'can limit transactions'),
                ('emergencyWithdraw', 'can emergency withdraw'),
                ('rugPull', 'has rug pull function'),
                ('drain', 'has drain function')
            ]
            
            for func_name, risk in dangerous_functions:
                func_hash = Web3.keccak(text=f"{func_name}()")[:4].hex()
                if func_hash in bytecode_hex:
                    risks.append(risk)
            
            honeypot_patterns = [
                ('require(from == owner() || to == owner()', 'owner-only transfers'),
                ('tradingEnabled || from == owner()', 'trading can be disabled'),
                ('canTrade[from] && canTrade[to]', 'whitelist-only trading')
            ]
            
            for pattern, risk in honeypot_patterns:
                pattern_hash = Web3.keccak(text=pattern)[:8].hex()
                if pattern_hash in bytecode_hex:
                    risks.append(risk)
                    
            token_info = await self.get_token_info(token_address, chain)
            if token_info:
                if token_info.owner != '0x0000000000000000000000000000000000000000':
                    risks.append('has owner')
                if token_info.is_paused:
                    risks.append('is paused')
                if token_info.has_mint_function:
                    risks.append('can mint tokens')
                    
            return {
                'safe': len(risks) == 0,
                'risk_count': len(risks),
                'reasons': risks,
                'bytecode_size': len(bytecode_hex) // 2
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing contract {token_address}: {e}")
            return {'safe': False, 'reasons': ['Analysis failed']}

    async def simulate_sell(self, token_address: str, amount: int, chain: str) -> Dict[str, Any]:
        try:
            w3 = self.providers.get(chain)
            if not w3:
                return {'can_sell': False, 'reason': 'No provider'}
                
            token_info = await self.get_token_info(token_address, chain)
            if not token_info:
                return {'can_sell': False, 'reason': 'Invalid token'}
                
            weth_address = self.chains[chain]['weth']
            pair = await self.get_pair_info(token_address, weth_address, chain)
            
            if not pair:
                return {'can_sell': False, 'reason': 'No liquidity pool'}
                
            if pair.token0.lower() == token_address.lower():
                token_reserve = pair.reserve0
                weth_reserve = pair.reserve1
            else:
                token_reserve = pair.reserve1
                weth_reserve = pair.reserve0
                
            if token_reserve == 0 or amount >= token_reserve:
                return {'can_sell': False, 'reason': 'Insufficient liquidity'}
                
            amount_out = (amount * weth_reserve) // (token_reserve + amount)
            slippage = (amount / token_reserve) * 100
            
            return {
                'can_sell': True,
                'amount_out': amount_out,
                'slippage': slippage,
                'price_impact': slippage
            }
            
        except Exception as e:
            return {'can_sell': False, 'reason': str(e)}

    async def get_transaction_count(self, address: str, chain: str) -> int:
        w3 = self.providers.get(chain)
        if not w3:
            return 0
        try:
            return w3.eth.get_transaction_count(to_checksum_address(address))
        except:
            return 0

    async def close(self):
        if self.session:
            await self.session.close()

web3_manager = Web3Manager()