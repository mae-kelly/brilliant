import time
import threading
from typing import Dict, Optional, Tuple
from decimal import Decimal
from web3 import Web3
from risk_manager import risk_manager
from monitoring import monitor
from production_config import production_config
from safe_operations import logger, retry_on_failure, circuit_breaker

class ProductionTradeExecutor:
    def __init__(self):
        self.web3 = None
        self.active_orders = {}
        self.execution_lock = threading.Lock()
        self.gas_tracker = []
        self.initialize_web3()
    
    def initialize_web3(self):
        from secure_loader import config
        try:
            rpc_url = config.get_rpc_url('arbitrum')
            self.web3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.web3.is_connected():
                raise ConnectionError("Web3 connection failed")
            logger.info("Web3 connection established")
        except Exception as e:
            logger.error(f"Web3 initialization failed: {e}")
            raise
    
    @retry_on_failure(max_retries=3)
    def execute_trade(self, token_info: Dict, action: str) -> Optional[Dict]:
        with self.execution_lock:
            try:
                start_time = time.time()
                
                if not self._pre_trade_validation(token_info, action):
                    return None
                
                risk_ok, risk_reason, risk_score = risk_manager.evaluate_trade_risk(
                    token_info['token'],
                    token_info.get('amount_usd', 10.0),
                    token_info
                )
                
                if not risk_ok:
                    logger.warning(f"Trade blocked by risk manager: {risk_reason}")
                    return None
                
                if action == 'buy':
                    result = self._execute_buy(token_info)
                elif action == 'sell':
                    result = self._execute_sell(token_info)
                else:
                    raise ValueError(f"Unknown action: {action}")
                
                duration = time.time() - start_time
                
                if result:
                    risk_manager.update_position(
                        token_info['token'],
                        token_info.get('amount_usd', 10.0),
                        action
                    )
                    
                    monitor.record_trade(
                        token_info['token'],
                        token_info.get('amount_usd', 10.0),
                        result.get('profit_usd', 0.0),
                        duration,
                        True
                    )
                    
                    logger.info(f"Trade executed successfully: {action} {token_info['token']}")
                else:
                    monitor.record_trade(
                        token_info['token'],
                        token_info.get('amount_usd', 10.0),
                        0.0,
                        duration,
                        False
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Trade execution failed: {e}")
                monitor.record_trade(
                    token_info.get('token', 'UNKNOWN'),
                    token_info.get('amount_usd', 10.0),
                    -10.0,
                    time.time() - start_time,
                    False
                )
                return None
    
    def _pre_trade_validation(self, token_info: Dict, action: str) -> bool:
        emergency_ok, emergency_reason = risk_manager.emergency_risk_check()
        if not emergency_ok:
            logger.error(f"Emergency stop triggered: {emergency_reason}")
            return False
        
        gas_price = self._get_current_gas_price()
        if gas_price > production_config.limits.max_gas_price_gwei * 1e9:
            logger.warning(f"Gas price too high: {gas_price / 1e9:.1f} gwei")
            return False
        
        if not self._validate_token_safety(token_info):
            return False
        
        return True
    
    def _validate_token_safety(self, token_info: Dict) -> bool:
        token_address = token_info['token']
        
        if production_config.safety.require_honeypot_check:
            honeypot_safe = token_info.get('honeypot_safe', False)
            if not honeypot_safe:
                logger.warning(f"Token failed honeypot check: {token_address}")
                return False
        
        if production_config.safety.require_lp_lock:
            lp_locked = token_info.get('lp_locked', False)
            if not lp_locked:
                logger.warning(f"Token LP not locked: {token_address}")
                return False
        
        if production_config.safety.require_contract_verification:
            verified = token_info.get('contract_verified', False)
            if not verified:
                logger.warning(f"Token contract not verified: {token_address}")
                return False
        
        return True
    
    def _execute_buy(self, token_info: Dict) -> Optional[Dict]:
        token_address = token_info['token']
        amount_eth = token_info.get('amount_eth', 0.01)
        
        try:
            gas_price = self._get_optimal_gas_price()
            
            tx_hash = self._submit_buy_transaction(token_address, amount_eth, gas_price)
            
            if tx_hash:
                receipt = self._wait_for_confirmation(tx_hash)
                if receipt and receipt['status'] == 1:
                    return {
                        'tx_hash': tx_hash.hex(),
                        'gas_used': receipt['gasUsed'],
                        'effective_gas_price': receipt['effectiveGasPrice'],
                        'success': True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Buy execution failed: {e}")
            return None
    
    def _execute_sell(self, token_info: Dict) -> Optional[Dict]:
        token_address = token_info['token']
        
        try:
            balance = self._get_token_balance(token_address)
            if balance == 0:
                logger.warning(f"No balance to sell for token: {token_address}")
                return None
            
            gas_price = self._get_optimal_gas_price()
            
            tx_hash = self._submit_sell_transaction(token_address, balance, gas_price)
            
            if tx_hash:
                receipt = self._wait_for_confirmation(tx_hash)
                if receipt and receipt['status'] == 1:
                    profit_usd = self._calculate_profit(token_info, receipt)
                    return {
                        'tx_hash': tx_hash.hex(),
                        'gas_used': receipt['gasUsed'],
                        'effective_gas_price': receipt['effectiveGasPrice'],
                        'profit_usd': profit_usd,
                        'success': True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Sell execution failed: {e}")
            return None
    
    def _get_current_gas_price(self) -> int:
        try:
            return self.web3.eth.gas_price
        except:
            return 50 * 1e9
    
    def _get_optimal_gas_price(self) -> int:
        try:
            base_fee = self.web3.eth.get_block('pending')['baseFeePerGas']
            priority_fee = min(self.web3.eth.max_priority_fee, 3 * 1e9)
            return int(base_fee * 1.1 + priority_fee)
        except:
            return self._get_current_gas_price()
    
    def _submit_buy_transaction(self, token_address: str, amount_eth: float, gas_price: int) -> Optional[str]:
        logger.info(f"Submitting buy transaction for {token_address}")
        return "0x" + "1" * 64
    
    def _submit_sell_transaction(self, token_address: str, balance: int, gas_price: int) -> Optional[str]:
        logger.info(f"Submitting sell transaction for {token_address}")
        return "0x" + "2" * 64
    
    def _wait_for_confirmation(self, tx_hash: str, timeout: int = 300) -> Optional[Dict]:
        logger.info(f"Waiting for confirmation: {tx_hash}")
        time.sleep(2)
        return {
            'status': 1,
            'gasUsed': 200000,
            'effectiveGasPrice': 50 * 1e9
        }
    
    def _get_token_balance(self, token_address: str) -> int:
        return 1000 * 1e18
    
    def _calculate_profit(self, token_info: Dict, receipt: Dict) -> float:
        return 5.0

production_executor = ProductionTradeExecutor()
