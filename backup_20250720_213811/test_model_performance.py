import pytest
import time
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.models.inference_model import backup_20250720_213811.inference_model
from core.models.model_manager import backup_20250720_213811.model_manager
from intelligence.analysis.advanced_ensemble import intelligence.analysis.advanced_ensemble

class TestModelPerformance:
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample features for testing"""
        return pd.DataFrame({
            'returns': np.random.normal(0, 0.01, 1000),
            'volatility': np.random.uniform(0.1, 0.3, 1000),
            'momentum': np.random.normal(0, 0.05, 1000),
            'rsi': np.random.uniform(30, 70, 1000),
            'bb_position': np.random.uniform(0, 1, 1000),
            'volume_ma': np.random.uniform(1000, 10000, 1000),
            'whale_activity': np.random.uniform(0, 0.2, 1000),
            'price_acceleration': np.random.normal(0, 0.001, 1000),
            'volatility_ratio': np.random.uniform(0.8, 1.2, 1000),
            'momentum_strength': np.random.uniform(0, 0.1, 1000),
            'swap_volume': np.random.uniform(1000, 10000, 1000)
        })
    
    def test_model_inference_latency(self, sample_features):
        """Test model inference meets latency requirements (<100ms)"""
        model = MomentumEnsemble()
        
        # Warmup
        for _ in range(10):
            model.predict(sample_features.tail(1))
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            model.predict(sample_features.tail(1))
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        p95_time = np.percentile(times, 95) * 1000
        
        print(f"Average inference time: {avg_time:.2f}ms")
        print(f"P95 inference time: {p95_time:.2f}ms")
        
        # Requirements
        assert avg_time < 100, f"Average inference time {avg_time:.2f}ms exceeds 100ms limit"
        assert p95_time < 200, f"P95 inference time {p95_time:.2f}ms exceeds 200ms limit"
    
    def test_batch_inference_throughput(self, sample_features):
        """Test batch inference throughput (>1000 predictions/sec)"""
        model = MomentumEnsemble()
        
        batch_size = 100
        num_batches = 10
        
        start_time = time.time()
        
        for _ in range(num_batches):
            batch_data = sample_features.sample(batch_size)
            for i in range(batch_size):
                model.predict(batch_data.iloc[i:i+1])
        
        end_time = time.time()
        
        total_predictions = batch_size * num_batches
        throughput = total_predictions / (end_time - start_time)
        
        print(f"Throughput: {throughput:.0f} predictions/second")
        
        assert throughput > 1000, f"Throughput {throughput:.0f} predictions/sec below 1000 req"
    
    @pytest.mark.asyncio
    async def test_advanced_ensemble_performance(self, sample_features):
        """Test advanced ensemble model performance"""
        model = AdvancedEnsembleModel()
        
        # Test single prediction
        start_time = time.time()
        
        result = await model.predict_with_multi_modal(
            'arbitrum', 
            '0x1234567890123456789012345678901234567890',
            sample_features.tail(1),
            'TEST'
        )
        
        end_time = time.time()
        prediction_time = (end_time - start_time) * 1000
        
        print(f"Advanced ensemble prediction time: {prediction_time:.2f}ms")
        
        # Verify result structure
        assert 'ensemble_prediction' in result
        assert 'confidence' in result
        assert 'uncertainty' in result
        assert isinstance(result['ensemble_prediction'], float)
        assert 0 <= result['ensemble_prediction'] <= 1
        
        # Performance requirement for production
        assert prediction_time < 5000, f"Ensemble prediction time {prediction_time:.2f}ms exceeds 5s limit"
    
    def test_memory_usage(self, sample_features):
        """Test memory usage remains reasonable"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = MomentumEnsemble()
        
        # Run many predictions
        for _ in range(1000):
            model.predict(sample_features.sample(1))
        
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Should not increase memory by more than 500MB
        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB too high"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
