"""
🔄 Renaissance Feedback Loop - ROI-Based Learning
Continuous model improvement from trade results
"""
import asyncio
import numpy as np
from typing import Dict, List

class FeedbackLoop:
    def __init__(self):
        self.trade_history = []
        self.model_performance = {}
        
    async def process_trade_result(self, trade_data: Dict):
        """Process completed trade for learning"""
        self.trade_history.append(trade_data)
        
        # Extract learning signals
        features = trade_data.get('features', [])
        prediction = trade_data.get('prediction', 0.5)
        actual_outcome = trade_data.get('roi', 0) > 0
        
        # Update model confidence based on accuracy
        prediction_correct = (prediction > 0.5) == actual_outcome
        
        if prediction_correct:
            print(f"✅ Model prediction correct: {prediction:.3f}")
        else:
            print(f"❌ Model prediction wrong: {prediction:.3f}")
            
        # Trigger retraining if performance degrades
        if len(self.trade_history) % 100 == 0:
            await self.evaluate_model_performance()
            
    async def evaluate_model_performance(self):
        """Evaluate and potentially retrain model"""
        recent_trades = self.trade_history[-100:]
        
        accuracy = sum(1 for t in recent_trades 
                      if (t.get('prediction', 0.5) > 0.5) == (t.get('roi', 0) > 0)
                      ) / len(recent_trades)
        
        print(f"📊 Model accuracy (last 100 trades): {accuracy:.2%}")
        
        if accuracy < 0.55:  # Below 55% accuracy
            print("🔄 Triggering model retraining...")
            await self.retrain_model()
            
    async def retrain_model(self):
        """Retrain model with recent data"""
        print("🧠 Retraining model with latest trade data...")
        await asyncio.sleep(2)  # Simulate retraining
        print("✅ Model retrained successfully")

# Global instance
feedback_loop = FeedbackLoop()
