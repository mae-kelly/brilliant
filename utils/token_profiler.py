from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

import json
import time
import requests
from statistics import mean
from web3 import Web3
from eth_abi import decode_abi


class TokenProfiler:
    def __init__(self, rpc_url, gecko_api_key=None, etherscan_api_key=None):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise Exception("Web3 connection failed.")

        self.gecko_api_key = gecko_api_key
        self.etherscan_api_key = etherscan_api_key
        self.erc20_abi = self._load_abi("/mnt/data/abi/erc20.json")
        self.proxy_abi = self._load_abi("/mnt/data/abi/proxy_checker.json")

    def _load_abi(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_token_contract(self, token_address):
        return self.web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=self.erc20_abi)

    def fetch_metadata(self, *args, **kwargs):
        try:
            token = self.get_token_contract(token_address)
            name = token.functions.name().call()
            symbol = token.functions.symbol().call()
            decimals = token.functions.decimals().call()
            total_supply = token.functions.totalSupply().call() / (10 ** decimals)
            return {
                "name": name,
                "symbol": symbol,
                "decimals": decimals,
                "total_supply": total_supply
            }
except Exception as e:
    logger.error(f"Error in operation: {e}")
    continue
            return None
        except:
            return None

    def whale_concentration_score(self, *args, **kwargs):
        try:
            url = f"https://api.etherscan.io/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset={top_n}&apikey={self.etherscan_api_key}"
            r = requests.get(url, timeout=10)
            holders = r.json()["result"]
            shares = [float(h["TokenHolderQuantity"]) for h in holders]
            total = sum(shares)
            top_pct = sum(shares[:top_n]) / total if total > 0 else 0
            return round(top_pct * 100, 2)
        except:
            return None

    def calculate_entropy_score(self, metadata):
        score = 100
        if metadata.get("total_supply", 0) < 1_000_000:
            score -= 10
        if len(metadata.get("symbol", "")) > 6:
            score -= 5
        if not metadata.get("name", "").isalpha():
            score -= 5
        return max(0, score)

    def calculate_risk_score(self, profile):
        score = 100
        if not profile.get("verified"):
            score -= 30
        if profile.get("is_proxy"):
            score -= 15
        if not profile.get("honeypot_safe"):
            score -= 40
        if profile.get("whale_pct", 0) > 50:
            score -= 25
        return max(0, score)

    def full_profile(self, token_address):
        token_address = Web3.to_checksum_address(token_address)
        profile = {"token_address": token_address}

        metadata = self.fetch_metadata(token_address)
        profile.update(metadata)

        profile["verified"] = self.check_verification_status(token_address)
        profile["is_proxy"] = self.is_proxy_contract(token_address)
        profile["ownership"] = self.assess_owner_permissions(token_address)
        profile.update(self.get_creation_details(token_address))
        profile["social_scores"] = self.get_social_metrics(token_address)
        profile["whale_pct"] = self.whale_concentration_score(token_address)
        profile["honeypot_safe"] = self.detect_honeypot_behavior(token_address, 10**15)
        profile["entropy_score"] = self.calculate_entropy_score(metadata)
        profile["risk_score"] = self.calculate_risk_score(profile)

        return profile

    def batch_profile(self, tokens):
        results = []
        for token in tokens:
            try:
                results.append(self.full_profile(token))
                time.sleep(0.5)
            except Exception as e:
                results.append({"token_address": token, "error": str(e)})
        return results
