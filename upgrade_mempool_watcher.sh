#!/bin/bash
cat > realtime_mempool_watcher.py << 'INNEREOF'
import asyncio
import websockets
import json
import time
import numpy as np
from collections import deque, defaultdict
import os

class UltraFastMempoolWatcher:
    def __init__(self):
        self.ws_endpoints = [
            "wss://eth-mainnet.g.alchemy.com/v2/" + os.getenv("API_KEY", "demo"),
            "wss://arb-mainnet.g.alchemy.com/v2/" + os.getenv("API_KEY", "demo")
        ]
        
        self.pending_tx_queue = asyncio.Queue(maxsize=100000)
        self.gas_price_history = deque(maxlen=1000)
        self.mev_opportunity_signals = deque(maxlen=500)
        
    async def initialize_all_streams(self):
        tasks = []
        for endpoint in self.ws_endpoints:
            task = asyncio.create_task(self.maintain_websocket_stream(endpoint))
            tasks.append(task)
            
        tasks.append(asyncio.create_task(self.process_pending_transactions()))
        
        return tasks
        
    async def maintain_websocket_stream(self, endpoint):
        while True:
            try:
                await asyncio.sleep(1)
            except:
                await asyncio.sleep(5)
                
    async def process_pending_transactions(self):
        while True:
            try:
                await asyncio.sleep(0.01)
            except:
                await asyncio.sleep(1)

watcher = UltraFastMempoolWatcher()
INNEREOF
echo "✅ Mempool watcher created"
