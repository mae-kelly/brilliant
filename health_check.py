from flask import Flask, jsonify
import psutil
import time
from monitoring import monitor
from risk_manager import risk_manager
from production_config import production_config

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime': time.time() - monitor.start_time
    })

@app.route('/metrics')
def metrics():
    performance = monitor.get_performance_report()
    risk_metrics = risk_manager.get_risk_report()
    
    return jsonify({
        'performance': performance.__dict__,
        'risk': risk_metrics.__dict__,
        'system': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    })

@app.route('/emergency-stop', methods=['POST'])
def emergency_stop():
    risk_manager.emergency_stops += 1
    return jsonify({'status': 'emergency stop activated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
