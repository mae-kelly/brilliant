import os
import json
import time
import requests
from web3 import Web3
from eth_abi import decode_abi


class AntiRugAnalyzer:
    def __init__(self, rpc_url, etherscan_api_key=None, gecko_api_key=None):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise Exception("Web3 connection failed.")
        self.etherscan_api_key = etherscan_api_key
        self.gecko_api_key = gecko_api_key
        self.erc20_abi = self._load_abi("/mnt/data/abi/erc20.json")

    def _load_abi(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def fetch_contract_source(self, *args, **kwargs):
        try:
            url = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={address}&apikey={self.etherscan_api_key}"
            r = requests.get(url, timeout=10)
            result = r.json()["result"][0]
            return result.get("SourceCode", ""), result.get("ABI", "")
        except:
            return "", ""

    def detect_mint_functions(self, *args, **kwargs):
        try:
            parsed = json.loads(abi)
            for func in parsed:
                if func.get("type") == "function" and "mint" in func.get("name", "").lower():
                    return True
            return False
        except:
            return False

    def detect_blacklist_or_pause(self, *args, **kwargs):
        try:
            parsed = json.loads(abi)
            for func in parsed:
                name = func.get("name", "").lower()
                if any(x in name for x in ["pause", "blacklist", "stop", "disable"]):
                    return True
            return False
        except:
            return False

    def check_lp_locked(self, *args, **kwargs):
        try:
            # Use GeckoTerminal API or similar for LP lock info
            if not self.gecko_api_key:
                return False
            headers = {"x-api-key": self.gecko_api_key}
            url = f"https://api.geckoterminal.com/api/v2/networks/eth/tokens/{lp_token_address}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return False
            data = r.json()["data"]["attributes"]
            return data.get("liquidity_locked", False)
        except:
            return False

    def detect_liquidity_removal_patterns(self, *args, **kwargs):
        try:
            pair_contract = self.web3.eth.contract(address=pair_address, abi=self.erc20_abi)
            events = pair_contract.events.Transfer.create_filter(fromBlock='latest').get_all_entries()
            liquidity_removed = 0
            for e in events:
                if e["args"]["to"] in [os.getenv("WALLET_ADDRESS", "os.getenv("WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")")]:
                    liquidity_removed += e["args"]["value"]
            return liquidity_removed > 0
        except:
            return False

    def decode_constructor(self, *args, **kwargs):
        try:
            if bytecode.startswith("0x"):
                bytecode = bytecode[2:]
            if "608060" in bytecode:  # Most contracts start with this
                offset = bytecode.find("608060")
                return bytecode[offset:]
            return bytecode
        except:
            return ""

    def check_liquidity_addition_block(self, *args, **kwargs):
        try:
            url = f"https://api.etherscan.io/api?module=account&action=tokentx&contractaddress={token_address}&startblock=0&endblock=99999999&sort=asc&apikey={self.etherscan_api_key}"
            r = requests.get(url, timeout=10)
            result = r.json()["result"]
            if not result:
                return None
            for tx in result:
                if int(tx["tokenDecimal"]) >= 6 and int(tx["value"]) > 0:
                    return int(tx["blockNumber"])
            return None
        except:
            return None

    def assess_contract_entropy(self, *args, **kwargs):
        try:
            if bytecode.startswith("0x"):
                bytecode = bytecode[2:]
            chunks = [bytecode[i:i + 2] for i in range(0, len(bytecode), 2)]
            unique = len(set(chunks))
            return round(unique / len(chunks), 4)
        except:
            return 0

    def detect_rebase_logic(self, *args, **kwargs):
        try:
            parsed = json.loads(abi)
            for fn in parsed:
                if fn.get("type") == "function" and any(x in fn.get("name", "").lower() for x in ["rebase", "adjustSupply"]):
                    return True
            return False
        except:
            return False

    def deep_rug_profile(self, token_address, pair_address=None):
        token_address = Web3.to_checksum_address(token_address)
        bytecode = self.web3.eth.get_code(token_address).hex()
        source_code, abi = self.fetch_contract_source(token_address)

        profile = {
            "token_address": token_address,
            "mint_detected": self.detect_mint_functions(abi),
            "blacklist_pause_detected": self.detect_blacklist_or_pause(abi),
            "contract_entropy": self.assess_contract_entropy(bytecode),
            "verified_source": bool(source_code),
            "rebase_logic": self.detect_rebase_logic(abi),
            "honeypot_safe": "transfer" in abi and "approve" in abi
        }

        if pair_address:
            profile.update({
                "lp_locked": self.check_lp_locked(pair_address),
                "liquidity_removed_recently": self.detect_liquidity_removal_patterns(pair_address)
            })

        liquidity_block = self.check_liquidity_addition_block(token_address)
        profile["initial_liquidity_block"] = liquidity_block

        # Risk score
        score = 100
        if profile["mint_detected"]:
            score -= 25
        if profile["blacklist_pause_detected"]:
            score -= 20
        if profile["contract_entropy"] < 0.8:
            score -= 10
        if not profile["verified_source"]:
            score -= 15
        if profile.get("liquidity_removed_recently"):
            score -= 20
        if not profile.get("lp_locked"):
            score -= 10
        if profile["rebase_logic"]:
            score -= 10

        profile["rug_risk_score"] = max(0, score)

        return profile

    def batch_analyze(self, token_pair_tuples):
        results = []
        for token, pair in token_pair_tuples:
            try:
                results.append(self.deep_rug_profile(token, pair))
                time.sleep(0.3)
            except Exception as e:
                results.append({"token": token, "pair": pair, "error": str(e)})
        return results
