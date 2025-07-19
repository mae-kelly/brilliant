from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

import os
import asyncio
import json
import time
import threading
from web3 import Web3, WebsocketProvider, HTTPProvider
from hexbytes import HexBytes
from eth_abi import decode_abi
from eth_utils import keccak, to_checksum_address
from utils.token_profiler import extract_token_address_from_input
from utils.honeypot_detector import is_potential_swap_call
from utils.dex_interface import fetch_token_metadata

# === Configuration === #
INFURA_WS = "wss://mainnet.infura.io/ws/v3/" + os.getenv("API_KEY", "")"
INFURA_HTTP = "https://mainnet.infura.io/v3/" + os.getenv("API_KEY", "")"
FLASHBOTS_RELAY = "https://relay.flashbots.net"
HIGH_PRIORITY_MULTIPLIER = 1.75
GAS_SPIKE_THRESHOLD = 2.0
GAS_WINDOW_SIZE = 12
CACHE_EXPIRY = 120  # seconds

SWAP_SIGNATURES = {
    'swapExactETHForTokens': keccak(text='swapExactETHForTokens(uint256,address[],address,uint256)').hex()[:10],
    'swapETHForExactTokens': keccak(text='swapETHForExactTokens(uint256,address[],address,uint256)').hex()[:10],
    'swapExactTokensForETH': keccak(text='swapExactTokensForETH(uint256,uint256,address[],address,uint256)').hex()[:10],
    'swapTokensForExactETH': keccak(text='swapTokensForExactETH(uint256,uint256,address[],address,uint256)').hex()[:10],
    'swapExactTokensForTokens': keccak(text='swapExactTokensForTokens(uint256,uint256,address[],address,uint256)').hex()[:10],
    'swapTokensForExactTokens': keccak(text='swapTokensForExactTokens(uint256,uint256,address[],address,uint256)').hex()[:10],
}

WATCHED_SIGNATURES = set(SWAP_SIGNATURES.values())

class MempoolWatcher:
    def __init__(self, ws_url=INFURA_WS, http_url=INFURA_HTTP):
        self.w3_ws = Web3(WebsocketProvider(ws_url))
        self.w3_http = Web3(HTTPProvider(http_url))
        self.token_alert_cache = {}
        self.token_blacklist = set()
        self.gas_history = []

    def _extract_method_id(self, input_data):
        return input_data[:10]

    def _decode_swap_token(self, *args, **kwargs):
        try:
            decoded = decode_abi(['uint256', 'address[]', 'address', 'uint256'], HexBytes(input_data[10:]))
            path = decoded[1]
            if len(path) >= 2:
                return to_checksum_address(path[-1])
except Exception as e:
    logger.error(f"Error in operation: {e}")
    continue
                if self._is_relevant_tx(tx):
                    token_addr = self._decode_swap_token(tx['input'])
                    if not token_addr or token_addr in self.token_blacklist:
                        continue
                    score = self._score_transaction(tx)
                    if score >= 2:
                        self._record_flagged_token(token_addr, score, tx['hash'], tx['from'], tx['gasPrice'])
                        print(f"[FLAGGED] {token_addr} | Score: {score} | Tx: {tx['hash'].hex()}")
                self._refresh_gas_window(tx['gasPrice'])
            except Exception as e:
    logger.error(f"Error: {e}")
    continue

    def blacklist_token(self, token_addr):
        self.token_blacklist.add(token_addr)

    def run_continuous(self):
        loop = threading.Thread(target=self.watch_pending_transactions)
        loop.start()

