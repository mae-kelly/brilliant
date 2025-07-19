from flask import Flask, render_template, jsonify
from safety_manager import safety_manager
import json

app = Flask(__name__)

@app.route('/safety')
def safety_dashboard():
    return jsonify({
        'emergency_stop': safety_manager.emergency_stop,
        'daily_loss': sum(safety_manager.daily_losses),
        'trades_last_hour': len(safety_manager.hourly_trades),
        'blacklisted_count': len(safety_manager.blacklisted_tokens),
        'recent_trades': list(safety_manager.trade_history)[-10:]
    })

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    safety_manager.emergency_shutdown("Manual trigger")
    return jsonify({'status': 'Emergency stop activated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
