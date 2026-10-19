"""
train_model.py
=============================================================
Smart Traffic Congestion Monitoring & Prediction System
B.Tech 2nd-Year CSE Mini-Project

This script trains a Logistic Regression classifier using
NumPy and Scikit-learn (without using Pandas).

Steps performed:
1. Load the simulated traffic dataset (traffic_data.csv) using Python's csv module.
2. One-hot encode categorical features (Road name and Day of week) with NumPy.
3. Combine with numerical features (Hour, Vehicle Count, Average Speed).
4. Split data into training (80%) and testing (20%) sets.
5. Standardize features using StandardScaler.
6. Train a Logistic Regression model for multi-class classification:
   - 0 = Low Congestion
   - 1 = Moderate Congestion
   - 2 = High Congestion
7. Evaluate model accuracy on the test set.
8. Save the trained model and scaler to disk for Flask to use.
=============================================================
"""

import csv
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# List of predefined roads in Mohali used in the project
ROAD_LIST = [
    'Airport Road',
    'Phase 7 Market Road',
    'Phase 3B2 Light Point',
    'Landran Road',
    'Kharar-Mohali Highway',
    'IT City Expressway',
    'Kumbra Chowk',
    'PCA Stadium Road'
]

# Days of the week in standard order
DAY_LIST = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]


def encode_features(road_name, day_of_week, hour, vehicle_count, avg_speed):
    """
    Converts raw input features into a numerical NumPy vector.
    - Road Name: One-Hot Encoded (vector of length 5)
    - Day of Week: One-Hot Encoded (vector of length 7)
    - Numerical values: [hour, vehicle_count, avg_speed] (length 3)
    Total feature vector length = 5 + 7 + 3 = 15 features.
    """
    # 1. One-hot encoding for road name
    road_vector = [1.0 if road_name == r else 0.0 for r in ROAD_LIST]

    # 2. One-hot encoding for day of week
    day_vector = [1.0 if day_of_week == d else 0.0 for d in DAY_LIST]

    # 3. Continuous numerical features
    numeric_vector = [float(hour), float(vehicle_count), float(avg_speed)]

    # Concatenate all features into a single NumPy array
    feature_vector = np.array(road_vector + day_vector + numeric_vector, dtype=np.float64)
    return feature_vector


def load_dataset(csv_filepath):
    """
    Loads traffic data from a CSV file using Python's built-in csv module.
    Converts rows into NumPy feature matrix X and label array y.
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Dataset file not found: {csv_filepath}")

    features_list = []
    labels_list = []

    with open(csv_filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vector = encode_features(
                row['road_name'].strip(),
                row['day_of_week'].strip(),
                row['hour_of_day'],
                row['vehicle_count'],
                row['avg_speed_kmh']
            )
            label = int(row['congestion_level'])
            features_list.append(vector)
            labels_list.append(label)

    # Convert Python lists to NumPy arrays
    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=np.int64)

    return X, y


def train():
    """
    Main training routine:
    Loads data, trains the model, prints metrics, and saves model files.
    """
    print("=" * 60)
    print(" Smart Traffic Congestion Prediction - Model Training")
    print("=" * 60)

    # Determine paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'traffic_data.csv')
    model_path = os.path.join(base_dir, 'model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')

    # Step 1: Load dataset
    print(f"\n[1/5] Loading dataset from: {csv_path}")
    X, y = load_dataset(csv_path)
    print(f"      Total samples loaded: {X.shape[0]}")
    print(f"      Feature dimensions per sample: {X.shape[1]}")

    # Step 2: Split data into training and testing sets (80% train, 20% test)
    print("\n[2/5] Splitting dataset (80% Training, 20% Testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training samples: {X_train.shape[0]}")
    print(f"      Testing samples:  {X_test.shape[0]}")

    # Step 3: Feature Scaling with StandardScaler
    print("\n[3/5] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Train Logistic Regression Classifier
    print("\n[4/5] Training Logistic Regression multi-class classifier...")
    # Multi-class Logistic Regression with L2 regularization
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Step 5: Evaluate model performance on the test set
    print("\n[5/5] Evaluating model on the unseen test set...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"      Test Set Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    target_names = ['Low Congestion (0)', 'Moderate Congestion (1)', 'High Congestion (2)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Important Academic Disclaimer
    print("-" * 60)
    print("ACADEMIC NOTE:")
    print("This model is trained on a simulated dataset designed for academic")
    print("demonstration. The high test accuracy verifies that the model has")
    print("correctly learned the classification boundaries from the simulated")
    print("rules, but does not establish real-world live traffic accuracy.")
    print("-" * 60)

    # Save model and scaler using joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[OK] Trained model successfully saved to: {model_path}")
    print(f"[OK] Scaler successfully saved to:        {scaler_path}")
    print("=" * 60)


if __name__ == '__main__':
    train()

