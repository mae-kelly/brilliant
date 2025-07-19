
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import os
import json
import time
import hashlib
import shutil
from typing import Dict, List, Optional
import tensorflow as tf
import numpy as np
from datetime import datetime
import sqlite3

class ModelRegistry:
    def __init__(self, registry_path: str = 'models/registry.db'):
        self.registry_path = registry_path
        self.models_dir = 'models'
        self.active_model_path = os.path.join(self.models_dir, 'active_model.tflite')
        self.init_database()

    def init_database(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE,
                model_hash TEXT,
                file_path TEXT,
                performance_metrics TEXT,
                created_at TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE,
                validation_score REAL,
                production_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                model_a_version TEXT,
                model_b_version TEXT,
                traffic_split REAL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                winner_version TEXT,
                results TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                timestamp TIMESTAMP,
                prediction_accuracy REAL,
                roi_performance REAL,
                latency_ms REAL,
                trade_volume INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()

    def calculate_model_hash(self, model_path: str) -> str:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        return hashlib.sha256(model_data).hexdigest()

    def register_model(self, model_path: str, version: str, metrics: Dict) -> bool:
        model_hash = self.calculate_model_hash(model_path)
        
        versioned_path = os.path.join(self.models_dir, f'model_{version}.tflite')
        shutil.copy2(model_path, versioned_path)
        
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO models (version, model_hash, file_path, performance_metrics, 
                                  created_at, validation_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                version, model_hash, versioned_path, json.dumps(metrics),
                datetime.now(), metrics.get('validation_accuracy', 0.0)
            ))
            
            conn.commit()
            print(f"Model {version} registered successfully")
            return True
            
        except sqlite3.IntegrityError:
            print(f"Model version {version} already exists")
            return False
        finally:
            conn.close()

    def deploy_model(self, version: str) -> bool:
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT file_path FROM models WHERE version = ?', (version,))
        result = cursor.fetchone()
        
        if not result:
            print(f"Model version {version} not found")
            conn.close()
            return False
            
        model_path = result[0]
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            conn.close()
            return False
            
        shutil.copy2(model_path, self.active_model_path)
        
        cursor.execute('UPDATE models SET is_active = FALSE')
        cursor.execute('UPDATE models SET is_active = TRUE WHERE version = ?', (version,))
        
        conn.commit()
        conn.close()
        
        print(f"Model {version} deployed as active model")
        return True

    def start_ab_test(self, test_name: str, model_a: str, model_b: str, 
                     traffic_split: float = 0.5) -> bool:
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_tests (test_name, model_a_version, model_b_version, 
                                traffic_split, start_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (test_name, model_a, model_b, traffic_split, datetime.now()))
        
        conn.commit()
        conn.close()
        
        print(f"A/B test '{test_name}' started: {model_a} vs {model_b}")
        return True

    def get_ab_test_model(self, test_name: str) -> Optional[str]:
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_a_version, model_b_version, traffic_split 
            FROM ab_tests 
            WHERE test_name = ? AND end_time IS NULL
        ''', (test_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        model_a, model_b, split = result
        return model_a if np.random.random() < split else model_b

    def log_performance(self, model_version: str, accuracy: float, 
                       roi: float, latency: float, volume: int):
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_logs (model_version, timestamp, prediction_accuracy,
                                        roi_performance, latency_ms, trade_volume)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_version, datetime.now(), accuracy, roi, latency, volume))
        
        conn.commit()
        conn.close()

    def evaluate_ab_test(self, test_name: str) -> Dict:
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_a_version, model_b_version FROM ab_tests 
            WHERE test_name = ? AND end_time IS NULL
        ''', (test_name,))
        
        test_info = cursor.fetchone()
        if not test_info:
            conn.close()
            return {}
            
        model_a, model_b = test_info
        
        cursor.execute('''
            SELECT model_version, AVG(roi_performance), AVG(prediction_accuracy), COUNT(*)
            FROM performance_logs 
            WHERE model_version IN (?, ?) AND timestamp > (
                SELECT start_time FROM ab_tests WHERE test_name = ?
            )
            GROUP BY model_version
        ''', (model_a, model_b, test_name))
        
        results = cursor.fetchall()
        conn.close()
        
        performance = {}
        for version, avg_roi, avg_accuracy, count in results:
            performance[version] = {
                'avg_roi': avg_roi,
                'avg_accuracy': avg_accuracy,
                'sample_count': count
            }
            
        return performance

    def finalize_ab_test(self, test_name: str, winner_version: str) -> bool:
        results = self.evaluate_ab_test(test_name)
        
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE ab_tests 
            SET end_time = ?, winner_version = ?, results = ?
            WHERE test_name = ?
        ''', (datetime.now(), winner_version, json.dumps(results), test_name))
        
        conn.commit()
        conn.close()
        
        self.deploy_model(winner_version)
        print(f"A/B test '{test_name}' completed. Winner: {winner_version}")
        return True

    def auto_rollback_on_performance_drop(self, threshold: float = 0.05):
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_version, AVG(roi_performance)
            FROM performance_logs 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY model_version
            HAVING COUNT(*) >= 10
        ''', )
        
        current_performance = cursor.fetchone()
        
        if current_performance:
            current_version, current_roi = current_performance
            
            cursor.execute('''
                SELECT version FROM models 
                WHERE is_active = FALSE 
                ORDER BY created_at DESC LIMIT 1
            ''')
            
            previous_version = cursor.fetchone()
            
            if previous_version and current_roi < -threshold:
                print(f"Performance drop detected. Rolling back to {previous_version[0]}")
                self.deploy_model(previous_version[0])
                
        conn.close()

class AutoModelManager:
    def __init__(self):
        self.registry = ModelRegistry()
        self.current_test = None
        self.performance_window = []

    def monitor_and_manage(self):
        while True:
            self.registry.auto_rollback_on_performance_drop()
            
            if self.current_test:
                results = self.registry.evaluate_ab_test(self.current_test)
                
                if len(results) == 2:
                    versions = list(results.keys())
                    perf_a = results[versions[0]]['avg_roi']
                    perf_b = results[versions[1]]['avg_roi']
                    
                    if abs(perf_a - perf_b) > 0.02 and min(results[v]['sample_count'] for v in versions) > 100:
                        winner = versions[0] if perf_a > perf_b else versions[1]
                        self.registry.finalize_ab_test(self.current_test, winner)
                        self.current_test = None
                        
            time.sleep(300)

    def deploy_new_model_with_testing(self, model_path: str, version: str, metrics: Dict):
        if self.registry.register_model(model_path, version, metrics):
            
            if self.current_test:
                self.registry.finalize_ab_test(self.current_test, version)
                
            current_active = self.get_active_model_version()
            if current_active:
                test_name = f"test_{current_active}_vs_{version}"
                self.registry.start_ab_test(test_name, current_active, version)
                self.current_test = test_name
            else:
                self.registry.deploy_model(version)

    def get_active_model_version(self) -> Optional[str]:
        conn = sqlite3.connect(self.registry.registry_path)
        cursor = conn.cursor()
        cursor.execute('SELECT version FROM models WHERE is_active = TRUE')
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

model_manager = AutoModelManager()
