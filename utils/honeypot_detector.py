import json
import traceback
from web3 import Web3
from eth_abi import decode_abi
from hexbytes import HexBytes


class HoneypotDetector:
    def __init__(self, rpc_url, router_address, factory_address, weth_address, test_wallet):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.web3.is_connected(), "RPC Connection Failed"
        self.router_address = Web3.to_checksum_address(router_address)
        self.factory_address = Web3.to_checksum_address(factory_address)
        self.weth_address = Web3.to_checksum_address(weth_address)
        self.test_wallet = Web3.to_checksum_address(test_wallet)

        self.router_abi = self._load_abi("uniswap_router.json")
        self.erc20_abi = self._load_abi("erc20.json")
        self.router = self.web3.eth.contract(address=self.router_address, abi=self.router_abi)

    def _load_abi(self, filename):
        path = f"/mnt/data/abi/{filename}"
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load ABI: {filename} - {str(e)}")

    def _simulate_tx(self, *args, **kwargs):
        try:
            result = self.web3.eth.call(txn_dict, "latest")
            return result
        except Exception as e:
            raise RuntimeError(f"Transaction simulation failed: {str(e)}")

    def is_buyable(self, token_address, amount_in_wei=10**16):
        token_address = Web3.to_checksum_address(token_address)
        path = [self.weth_address, token_address]
        deadline = self.web3.eth.get_block('latest')['timestamp'] + 120

        try:
            txn = self.router.functions.swapExactETHForTokens(
                0, path, self.test_wallet, deadline
            ).build_transaction({
                'from': self.test_wallet,
                'value': amount_in_wei,
                'gas': 300000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.test_wallet)
            })

            self._simulate_tx(txn)
            return True
        except Exception as e:
            print(f"[HONEYPOT-DETECT] Buy failed: {str(e)}")
            return False

    def is_sellable(self, token_address, test_amount_out=10**10):
        token_address = Web3.to_checksum_address(token_address)
        path = [token_address, self.weth_address]
        deadline = self.web3.eth.get_block('latest')['timestamp'] + 120

        try:
            token = self.web3.eth.contract(address=token_address, abi=self.erc20_abi)

            # Approve for swap
            approve_txn = token.functions.approve(
                self.router_address, test_amount_out
            ).build_transaction({
                'from': self.test_wallet,
                'gas': 100000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.test_wallet)
            })
            self._simulate_tx(approve_txn)

            # Attempt simulated sell
            swap_txn = self.router.functions.swapExactTokensForETH(
                test_amount_out, 0, path, self.test_wallet, deadline
            ).build_transaction({
                'from': self.test_wallet,
                'gas': 400000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.test_wallet) + 1
            })
            self._simulate_tx(swap_txn)
            return True
        except Exception as e:
            print(f"[HONEYPOT-DETECT] Sell failed: {str(e)}")
            return False

    def is_transferable(self, token_address):
        token_address = Web3.to_checksum_address(token_address)
        try:
            token = self.web3.eth.contract(address=token_address, abi=self.erc20_abi)
            tx = token.functions.transfer(
                self.test_wallet, 1
            ).build_transaction({
                'from': self.test_wallet,
                'nonce': self.web3.eth.get_transaction_count(self.test_wallet),
                'gas': 50000,
                'gasPrice': self.web3.eth.gas_price
            })
            self._simulate_tx(tx)
            return True
        except Exception as e:
            print(f"[HONEYPOT-DETECT] Transfer test failed: {str(e)}")
            return False

    def check_tokenomics_flags(self, *args, **kwargs):
        try:
            token = self.web3.eth.contract(address=token_address, abi=self.erc20_abi)
            total_supply = token.functions.totalSupply().call()
            name = token.functions.name().call()
            symbol = token.functions.symbol().call()
            decimals = token.functions.decimals().call()
            return {
                "supply": total_supply,
                "name": name,
                "symbol": symbol,
                "decimals": decimals
            }
        except:
            return {
                "supply": -1,
                "name": "UNKNOWN",
                "symbol": "??",
                "decimals": -1
            }

    def is_lp_locked(self, token_address):
        # Could integrate with services like Unicrypt, Deeplock, or check bytecode for LP time locks
        return "unknown"

    def full_analysis(self, *args, **kwargs):
        try:
            print(f"--- Honeypot Check: {token_address} ---")
            transferable = self.is_transferable(token_address)
            buyable = self.is_buyable(token_address)
            sellable = self.is_sellable(token_address)
            tokenomics = self.check_tokenomics_flags(token_address)
            lp_status = self.is_lp_locked(token_address)

            result = {
                "buyable": buyable,
                "sellable": sellable,
                "transferable": transferable,
                "tokenomics": tokenomics,
                "lp_locked": lp_status,
                "verdict": buyable and sellable and transferable
            }

            print(f"[RESULT] Verdict: {'PASS' if result['verdict'] else 'FAIL'}")
            return result
        except Exception as e:
            print(f"[HONEYPOT-DETECT] Exception: {str(e)}")
            traceback.print_exc()
            return {
                "buyable": False,
                "sellable": False,
                "transferable": False,
                "tokenomics": {},
                "lp_locked": False,
                "verdict": False
            }
