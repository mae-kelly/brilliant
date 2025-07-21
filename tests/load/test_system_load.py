import pytest
import asyncio
import aiohttp
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestSystemLoad:
    
    @pytest.mark.asyncio
    async def test_api_endpoint_load(self):
        """Test API endpoints under load"""
        
        # Start the FastAPI server (assuming it's running)
        base_url = "http://localhost:8000"
        
        async def make_prediction_request(session, request_id):
            """Make a single prediction request"""
            features = {
                'returns': float(np.random.normal(0, 0.01)),
                'volatility': float(np.random.uniform(0.1, 0.3)),
                'momentum': float(np.random.normal(0, 0.05)),
                'rsi': float(np.random.uniform(30, 70)),
                'bb_position': float(np.random.uniform(0, 1)),
                'volume_ma': float(np.random.uniform(1000, 10000)),
                'whale_activity': float(np.random.uniform(0, 0.2)),
                'price_acceleration': float(np.random.normal(0, 0.001)),
                'volatility_ratio': float(np.random.uniform(0.8, 1.2)),
                'momentum_strength': float(np.random.uniform(0, 0.1)),
                'swap_volume': float(np.random.uniform(1000, 10000))
            }
            
            start_time = time.time()
            
            try:
                async with session.post(f"{base_url}/predict", json=features, timeout=10) as response:
                    result = await response.json()
                    end_time = time.time()
                    
                    return {
                        'request_id': request_id,
                        'status_code': response.status,
                        'response_time': end_time - start_time,
                        'success': response.status == 200 and 'prediction' in result
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    'request_id': request_id,
                    'status_code': 0,
                    'response_time': end_time - start_time,
                    'success': False,
                    'error': str(e)
                }
        
        # Load test parameters
        concurrent_requests = 50
        total_requests = 200
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint first
            try:
                async with session.get(f"{base_url}/health", timeout=5) as response:
                    if response.status != 200:
                        pytest.skip("API server not running or not healthy")
            except:
                pytest.skip("API server not accessible")
            
            # Run load test
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_request(request_id):
                async with semaphore:
                    return await make_prediction_request(session, request_id)
            
            start_time = time.time()
            
            tasks = [limited_request(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # Analyze results
            successful_requests = [r for r in results if r['success']]
            failed_requests = [r for r in results if not r['success']]
            
            response_times = [r['response_time'] for r in successful_requests]
            
            total_time = end_time - start_time
            throughput = len(successful_requests) / total_time
            
            success_rate = len(successful_requests) / total_requests
            avg_response_time = np.mean(response_times) if response_times else 0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0
            
            print(f"\\nLoad Test Results:")
            print(f"Total requests: {total_requests}")
            print(f"Successful requests: {len(successful_requests)}")
            print(f"Failed requests: {len(failed_requests)}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Throughput: {throughput:.1f} req/sec")
            print(f"Average response time: {avg_response_time*1000:.1f}ms")
            print(f"P95 response time: {p95_response_time*1000:.1f}ms")
            
            # Performance requirements
            assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
            assert throughput >= 10, f"Throughput {throughput:.1f} req/sec below 10 req/sec"
            assert avg_response_time < 1.0, f"Avg response time {avg_response_time:.3f}s above 1s"
    
    def test_concurrent_model_inference(self):
        """Test model under concurrent load"""
        from inference_model import MomentumEnsemble
        import pandas as pd
        
        model = MomentumEnsemble()
        
        # Generate test data
        def generate_features():
            return pd.DataFrame({
                'returns': [np.random.normal(0, 0.01)],
                'volatility': [np.random.uniform(0.1, 0.3)],
                'momentum': [np.random.normal(0, 0.05)],
                'rsi': [np.random.uniform(30, 70)],
                'bb_position': [np.random.uniform(0, 1)],
                'volume_ma': [np.random.uniform(1000, 10000)],
                'whale_activity': [np.random.uniform(0, 0.2)],
                'price_acceleration': [np.random.normal(0, 0.001)],
                'volatility_ratio': [np.random.uniform(0.8, 1.2)],
                'momentum_strength': [np.random.uniform(0, 0.1)],
                'swap_volume': [np.random.uniform(1000, 10000)]
            })
        
        def run_inference(worker_id, num_predictions):
            """Run inference in a thread"""
            results = []
            start_time = time.time()
            
            for i in range(num_predictions):
                features = generate_features()
                try:
                    prediction = model.predict(features)
                    results.append({
                        'worker_id': worker_id,
                        'prediction_id': i,
                        'prediction': prediction,
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'worker_id': worker_id,
                        'prediction_id': i,
                        'success': False,
                        'error': str(e)
                    })
            
            end_time = time.time()
            
            return {
                'worker_id': worker_id,
                'results': results,
                'duration': end_time - start_time,
                'successful_predictions': len([r for r in results if r['success']])
            }
        
        # Concurrent inference test
        num_workers = 10
        predictions_per_worker = 50
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.time()
            
            futures = [
                executor.submit(run_inference, worker_id, predictions_per_worker)
                for worker_id in range(num_workers)
            ]
            
            worker_results = [future.result() for future in futures]
            
            end_time = time.time()
        
        # Analyze results
        total_predictions = sum(r['successful_predictions'] for r in worker_results)
        total_time = end_time - start_time
        throughput = total_predictions / total_time
        
        all_successful = all(r['successful_predictions'] == predictions_per_worker for r in worker_results)
        
        print(f"\\nConcurrent Inference Test:")
        print(f"Workers: {num_workers}")
        print(f"Predictions per worker: {predictions_per_worker}")
        print(f"Total successful predictions: {total_predictions}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} predictions/sec")
        print(f"All workers successful: {all_successful}")
        
        # Performance requirements
        assert all_successful, "Some workers failed to complete all predictions"
        assert throughput > 100, f"Concurrent throughput {throughput:.1f} below 100 predictions/sec"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
