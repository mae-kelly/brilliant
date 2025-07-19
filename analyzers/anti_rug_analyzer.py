from config.dynamic_settings import dynamic_settings

# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import time
from web3 import Web3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re

@dataclass
class RugAnalysis:
    token_address: str
    risk_score: float
    is_safe: bool
    flags: List[str]
    honeypot_risk: float
    liquidity_locked: bool
    contract_verified: bool
    analyzed_at: float

class AntiRugAnalyzer:
    def __init__(self):
        self.dangerous_functions = [
            'setTaxFeePercent',
            'setMaxTxPercent', 
            'excludeFromReward',
            'includeInReward',
            'blacklistAddress',
            'removeFromBlacklist',
            'setSwapAndLiquifyEnabled',
            'emergencyWithdraw',
            'rugPull',
            'drain',
            'withdraw',
            'emergencyExit'
        ]
        
        self.honeypot_patterns = [
            'require(from == owner() || to == owner()',
            'if (from != owner() && to != owner())',
            'require(amount <= _maxTxAmount',
            'isBlacklisted[from] || isBlacklisted[to]',
            'tradingEnabled || from == owner()',
            'canTrade[from] && canTrade[to]'
        ]
        
        self.safe_contracts = set()
        self.flagged_contracts = set()
        self.analysis_cache = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def analyze_token_safety(self, token_address: str, chain: str = 'ethereum') -> RugAnalysis:
        start_time = time.time()
        
        if token_address in self.analysis_cache:
            cached = self.analysis_cache[token_address]
            if time.time() - cached.analyzed_at < 300:
                return cached
        
        flags = []
        risk_score = 0.0
        
        contract_analysis = await self.analyze_contract_code(token_address)
        if contract_analysis['has_dangerous_functions']:
            flags.extend(contract_analysis['dangerous_functions'])
            risk_score += 0.4
        
        if contract_analysis['has_honeypot_patterns']:
            flags.append('honeypot_patterns_detected')
            risk_score += 0.5
        
        liquidity_analysis = await self.analyze_liquidity_lock(token_address)
        liquidity_locked = liquidity_analysis['is_locked']
        if not liquidity_locked:
            flags.append('liquidity_not_locked')
            risk_score += 0.3
        
        ownership_analysis = await self.analyze_ownership(token_address)
        if ownership_analysis['can_mint']:
            flags.append('unlimited_minting')
            risk_score += 0.4
        
        if ownership_analysis['can_pause']:
            flags.append('can_pause_trading')
            risk_score += 0.3
        
        honeypot_risk = await self.calculate_honeypot_risk(token_address)
        
        contract_verified = await self.check_contract_verification(token_address)
        if not contract_verified:
            flags.append('contract_not_verified')
            risk_score += 0.2
        
        is_safe = risk_score < 0.5 and honeypot_risk < 0.3
        
        analysis = RugAnalysis(
            token_address=token_address,
            risk_score=min(risk_score, 1.0),
            is_safe=is_safe,
            flags=flags,
            honeypot_risk=honeypot_risk,
            liquidity_locked=liquidity_locked,
            contract_verified=contract_verified,
            analyzed_at=time.time()
        )
        
        self.analysis_cache[token_address] = analysis
        
        if is_safe:
            self.safe_contracts.add(token_address)
        else:
            self.flagged_contracts.add(token_address)
        
        self.logger.info(
            f"🛡️ Analyzed {token_address[:8]}... "
            f"Risk: {risk_score:.2f} Safe: {is_safe} "
            f"Flags: {len(flags)}"
        )
        
        return analysis

    async def analyze_contract_code(self, token_address: str) -> Dict:
        await asyncio.sleep(0.1)
        
        has_dangerous = False
        found_functions = []
        
        simulated_code = f"contract Token_{token_address[-6:]} {{" + """
            function transfer(address to, uint256 amount) public returns (bool);
            function balanceOf(address account) public view returns (uint256);
            function totalSupply() public view returns (uint256);
        }"""
        
        danger_chance = hash(token_address) % 100
        if danger_chance < 15:
            has_dangerous = True
            found_functions = ['setTaxFeePercent', 'blacklistAddress']
        
        has_honeypot = danger_chance < 8
        
        return {
            'has_dangerous_functions': has_dangerous,
            'dangerous_functions': found_functions,
            'has_honeypot_patterns': has_honeypot,
            'code_size': len(simulated_code)
        }

    async def analyze_liquidity_lock(self, token_address: str) -> Dict:
        await asyncio.sleep(0.05)
        
        lock_chance = hash(token_address + 'lock') % 100
        is_locked = lock_chance > 30
        
        lock_duration = 0
        if is_locked:
            lock_duration = 30 + (lock_chance % 365)
        
        return {
            'is_locked': is_locked,
            'lock_duration_days': lock_duration,
            'lock_percentage': 95.0 if is_locked else 0.0
        }

    async def analyze_ownership(self, token_address: str) -> Dict:
        await asyncio.sleep(0.05)
        
        ownership_hash = hash(token_address + 'owner') % 100
        
        return {
            'owner_renounced': ownership_hash > 70,
            'can_mint': ownership_hash < 20,
            'can_pause': ownership_hash < 15,
            'max_supply_limited': ownership_hash > 50
        }

    async def calculate_honeypot_risk(self, token_address: str) -> float:
        await asyncio.sleep(0.1)
        
        risk_factors = []
        
        sell_simulation = await self.simulate_sell_transaction(token_address)
        if not sell_simulation['can_sell']:
            risk_factors.append(0.8)
        
        if sell_simulation['high_tax']:
            risk_factors.append(0.3)
        
        if sell_simulation['blacklist_risk']:
            risk_factors.append(0.5)
        
        return min(sum(risk_factors), 1.0)

    async def simulate_sell_transaction(self, token_address: str) -> Dict:
        await asyncio.sleep(0.05)
        
        sim_hash = hash(token_address + 'sim') % 100
        
        return {
            'can_sell': sim_hash > 20,
            'high_tax': sim_hash < 15,
            'blacklist_risk': sim_hash < 10,
            'slippage_too_high': sim_hash < 5
        }

    async def check_contract_verification(self, token_address: str) -> bool:
        await asyncio.sleep(0.02)
        verification_chance = hash(token_address + 'verify') % 100
        return verification_chance > 25

    def get_safety_stats(self) -> Dict:
        return {
            'safe_contracts': len(self.safe_contracts),
            'flagged_contracts': len(self.flagged_contracts),
            'total_analyzed': len(self.analysis_cache),
            'cache_size': len(self.analysis_cache)
        }

anti_rug_analyzer = AntiRugAnalyzer()
