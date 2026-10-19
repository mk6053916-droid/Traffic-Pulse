"""
app.py
=============================================================
TrafficPulse — Smart Traffic Congestion Monitoring & Prediction System
B.Tech 2nd-Year CSE Mini-Project

Flask Web Application:
- Connects the HTML/CSS/JS frontend with the Scikit-learn ML model.
- Loads the trained model and scaler on startup.
- Serves the main SPA ('/') and the dedicated About page ('/about').
- Validates user input submitted via fetch/AJAX ('/predict').
- Serves dynamic aggregated dataset metrics ('/chart-data' and '/analytics').
- Provides health monitoring ('/health').
- Uses NumPy and Python's built-in csv module (does NOT use Pandas).
=============================================================
"""

import csv
import os
import sys
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from feature_config import ROADS, DAYS

app = Flask(__name__)

# Base directory for resolving file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'traffic_data.csv')

# Load model and scaler when the application starts
model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[Flask Startup] Successfully loaded model.pkl and scaler.pkl")
    else:
        print("[Flask Startup WARNING] Model or scaler files not found!")
        print("Please run 'python train_model.py' first to train and save the model.")
except Exception as e:
    print(f"[Flask Startup ERROR] Failed to load model files: {e}")


def encode_features(road_name, day_of_week, hour, vehicle_count, avg_speed):
    """
    Transforms single user input into an 18-dimensional NumPy vector,
    matching the feature structure used during model training.
    """
    road_vector = [1.0 if road_name == r else 0.0 for r in ROADS]
    day_vector = [1.0 if day_of_week == d else 0.0 for d in DAYS]
    numeric_vector = [float(hour), float(vehicle_count), float(avg_speed)]

    return np.array([road_vector + day_vector + numeric_vector], dtype=np.float64)


# ---------------------------------------------------------------------------
# Route 1: Home Page (Main Prediction Dashboard)
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    """Renders the main single-page traffic monitoring and prediction dashboard."""
    return render_template('index.html', roads=ROADS, days=DAYS)


# ---------------------------------------------------------------------------
# Route 2: Dedicated About Page
# ---------------------------------------------------------------------------
@app.route('/about')
def about():
    """Renders the comprehensive project explanation, methodology, and documentation page."""
    return render_template('about.html', roads=ROADS, days=DAYS)


# ---------------------------------------------------------------------------
# Route 3: Predict Congestion Endpoint (POST /predict)
# ---------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives JSON input from frontend, validates values,
    runs the Scikit-learn Logistic Regression model, and returns
    the predicted congestion category and calibrated probabilities as JSON.
    """
    if model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Model or scaler not loaded. Please train the model first by running python train_model.py.'
        }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No input data received.'}), 400

        # Handle both parameter naming conventions for compatibility
        road = data.get('road_name') or data.get('road')
        day = data.get('day_of_week') or data.get('day')
        hour_raw = data.get('hour_of_day') if data.get('hour_of_day') is not None else data.get('hour')
        vehicles_raw = data.get('vehicle_count')
        speed_raw = data.get('avg_speed_kmh') if data.get('avg_speed_kmh') is not None else data.get('avg_speed')

        # Validation 1: Required fields
        if not road or not day or hour_raw is None or vehicles_raw is None or speed_raw is None:
            return jsonify({'success': False, 'error': 'Please fill in all the form fields.'}), 400

        # Validation 2: Road check
        if road not in ROADS:
            return jsonify({'success': False, 'error': f'Invalid road selected: {road}', 'field': 'road_name'}), 400

        # Validation 3: Day check
        if day not in DAYS:
            return jsonify({'success': False, 'error': f'Invalid day selected: {day}', 'field': 'day_of_week'}), 400

        # Validation 4: Hour check (0 to 23)
        try:
            hour = int(hour_raw)
            if not (0 <= hour <= 23):
                return jsonify({'success': False, 'error': 'Hour of the day must be between 0 and 23.', 'field': 'hour_of_day'}), 400
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Hour must be an integer between 0 and 23.', 'field': 'hour_of_day'}), 400

        # Validation 5: Vehicle count check (1 to 4000)
        try:
            vehicles = int(vehicles_raw)
            if vehicles <= 0 or vehicles > 4000:
                return jsonify({'success': False, 'error': 'Number of vehicles must be a realistic positive number (1 - 4000).', 'field': 'vehicle_count'}), 400
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Number of vehicles must be a valid whole number.', 'field': 'vehicle_count'}), 400

        # Validation 6: Average speed check (1 to 150 km/h)
        try:
            speed = float(speed_raw)
            if speed <= 0 or speed > 150:
                return jsonify({'success': False, 'error': 'Average speed must be between 1 and 150 km/h.', 'field': 'avg_speed_kmh'}), 400
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Average speed must be a valid number in km/h.', 'field': 'avg_speed_kmh'}), 400

        # Preprocess features: One-hot + numeric -> StandardScale
        raw_features = encode_features(road, day, hour, vehicles, speed)
        scaled_features = scaler.transform(raw_features)

        # Predict class index (0 = Low, 1 = Moderate, 2 = High)
        prediction_class = int(model.predict(scaled_features)[0])

        # Get calibrated Softmax probabilities
        probabilities = model.predict_proba(scaled_features)[0]

        categories = {
            0: {
                'label': 'Low Congestion',
                'short': 'Low',
                'color_code': 'green',
                'badge_class': 'result-low',
                'advisory': 'Traffic is light and flowing freely. Good traveling conditions with negligible delays.'
            },
            1: {
                'label': 'Moderate Congestion',
                'short': 'Moderate',
                'color_code': 'orange',
                'badge_class': 'result-moderate',
                'advisory': 'Normal traffic density observed. Mild slowdowns around junctions; maintain safe following distance.'
            },
            2: {
                'label': 'High Congestion',
                'short': 'High',
                'color_code': 'red',
                'badge_class': 'result-high',
                'advisory': 'Heavy congestion detected with high vehicle density and slow speeds. Expect noticeable delays.'
            }
        }

        result_info = categories.get(prediction_class, {
            'label': 'Unknown',
            'short': 'Unknown',
            'color_code': 'gray',
            'badge_class': 'result-unknown',
            'advisory': 'Unable to determine status.'
        })

        return jsonify({
            'success': True,
            'prediction_code': prediction_class,
            'prediction': result_info['short'],
            'congestion_label': result_info['label'],
            'label': result_info['label'],
            'color': result_info['color_code'],
            'badge_class': result_info['badge_class'],
            'advisory': result_info['advisory'],
            'confidence': {
                'low': round(float(probabilities[0]) * 100, 1),
                'moderate': round(float(probabilities[1]) * 100, 1),
                'high': round(float(probabilities[2]) * 100, 1)
            },
            'probabilities': {
                'Low': round(float(probabilities[0]), 4),
                'Moderate': round(float(probabilities[1]), 4),
                'High': round(float(probabilities[2]), 4)
            },
            'input': {
                'road_name': road,
                'day_of_week': day,
                'hour_of_day': hour,
                'vehicle_count': vehicles,
                'avg_speed_kmh': speed
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# Route 4: Chart Data Endpoint (GET /chart-data)
# ---------------------------------------------------------------------------
@app.route('/chart-data')
def chart_data():
    """
    Computes summary metrics directly from traffic_data.csv using
    Python's standard csv module and NumPy.
    """
    if not os.path.exists(CSV_PATH):
        return jsonify({'success': False, 'error': 'Dataset not found.'}), 404

    low_vehicles, mod_vehicles, high_vehicles = [], [], []
    low_speeds, mod_speeds, high_speeds = [], [], []

    try:
        with open(CSV_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                level = int(row['congestion_level'])
                count = float(row['vehicle_count'])
                speed = float(row['avg_speed_kmh'])
                if level == 0:
                    low_vehicles.append(count)
                    low_speeds.append(speed)
                elif level == 1:
                    mod_vehicles.append(count)
                    mod_speeds.append(speed)
                elif level == 2:
                    high_vehicles.append(count)
                    high_speeds.append(speed)

        avg_low_v = round(float(np.mean(low_vehicles)), 1) if low_vehicles else 0.0
        avg_mod_v = round(float(np.mean(mod_vehicles)), 1) if mod_vehicles else 0.0
        avg_high_v = round(float(np.mean(high_vehicles)), 1) if high_vehicles else 0.0

        avg_low_s = round(float(np.mean(low_speeds)), 1) if low_speeds else 0.0
        avg_mod_s = round(float(np.mean(mod_speeds)), 1) if mod_speeds else 0.0
        avg_high_s = round(float(np.mean(high_speeds)), 1) if high_speeds else 0.0

        return jsonify({
            'success': True,
            'labels': ['Low Congestion', 'Moderate Congestion', 'High Congestion'],
            'data': [avg_low_v, avg_mod_v, avg_high_v],
            'avg_vehicles': [avg_low_v, avg_mod_v, avg_high_v],
            'avg_speeds': [avg_low_s, avg_mod_s, avg_high_s],
            'sample_counts': [len(low_vehicles), len(mod_vehicles), len(high_vehicles)]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Route 5: Analytics Endpoint (GET /analytics)
# ---------------------------------------------------------------------------
@app.route('/analytics')
def analytics():
    """Returns dataset aggregates and hourly trend analysis."""
    if not os.path.exists(CSV_PATH):
        return jsonify({'success': False, 'error': 'Dataset not found.'}), 404

    try:
        rows = []
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        total = len(rows)
        total_vehicles = 0.0
        total_speed = 0.0
        cong_vehicles = {"Low": 0.0, "Moderate": 0.0, "High": 0.0}
        cong_counts = {"Low": 0, "Moderate": 0, "High": 0}
        hour_vehicles = [0.0] * 24
        hour_counts = [0] * 24
        road_vehicles = {r: 0.0 for r in ROADS}
        road_speeds = {r: 0.0 for r in ROADS}
        road_counts = {r: 0 for r in ROADS}

        label_keys = ['Low', 'Moderate', 'High']

        for row in rows:
            vc = float(row['vehicle_count'])
            spd = float(row['avg_speed_kmh'])
            hr = int(row['hour_of_day'])
            road = row['road_name']
            level = int(row['congestion_level'])

            label = label_keys[level] if level in (0, 1, 2) else None

            total_vehicles += vc
            total_speed += spd

            if label:
                cong_vehicles[label] += vc
                cong_counts[label] += 1

            if 0 <= hr <= 23:
                hour_vehicles[hr] += vc
                hour_counts[hr] += 1

            if road in road_vehicles:
                road_vehicles[road] += vc
                road_speeds[road] += spd
                road_counts[road] += 1

        hourly_stats = [
            {
                "hour": h,
                "avg_vehicles": round(hour_vehicles[h] / hour_counts[h], 1) if hour_counts[h] else 0.0
            }
            for h in range(24)
        ]

        congestion_stats = {
            k: round(cong_vehicles[k] / cong_counts[k], 1) if cong_counts[k] else 0.0
            for k in label_keys
        }

        road_stats = [
            {
                "road": r,
                "avg_vehicles": round(road_vehicles[r] / road_counts[r], 1) if road_counts[r] else 0.0,
                "avg_speed": round(road_speeds[r] / road_counts[r], 1) if road_counts[r] else 0.0,
                "count": road_counts[r]
            }
            for r in ROADS
        ]

        return jsonify({
            "success": True,
            "total_records": total,
            "road_count": len(ROADS),
            "avg_vehicle_count": round(total_vehicles / total, 1) if total else 0.0,
            "avg_speed": round(total_speed / total, 1) if total else 0.0,
            "congestion_stats": congestion_stats,
            "hourly_stats": hourly_stats,
            "road_stats": road_stats
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Route 6: Health Endpoint (GET /health)
# ---------------------------------------------------------------------------
@app.route('/health')
def health():
    """System diagnostic check confirming model, scaler, and dataset state."""
    dataset_exists = os.path.exists(CSV_PATH)
    total_records = 0
    if dataset_exists:
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            total_records = sum(1 for _ in csv.reader(f)) - 1

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "dataset_exists": dataset_exists,
        "total_records": total_records,
        "roads": len(ROADS)
    })


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"Starting TrafficPulse on http://127.0.0.1:{port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
