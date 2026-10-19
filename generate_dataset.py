"""
generate_dataset.py
-------------------
Generates a SIMULATED traffic dataset for the TrafficPulse project.
NOTE: This is NOT real Mohali traffic data. All values are synthetically
generated for educational and demonstration purposes only.
"""

import csv
import numpy
import random

# Fix random seed for reproducibility
numpy.random.seed(42)
random.seed(42)

from feature_config import ROADS, DAYS

OUTPUT_FILE = 'traffic_data.csv'
TOTAL_RECORDS = 800


def get_vehicle_count_for_hour(hour):
    """Return a vehicle count based on the time of day (rush/night/normal)."""
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        # Rush hours: higher traffic
        base_count = numpy.random.randint(200, 501)
    elif hour >= 23 or hour <= 5:
        # Night hours: very low traffic
        base_count = numpy.random.randint(20, 101)
    else:
        # Normal daytime hours: moderate traffic
        base_count = numpy.random.randint(100, 301)

    # Add ±15% noise to vehicle count
    noise_factor = 1.0 + numpy.random.uniform(-0.15, 0.15)
    noisy_count = int(base_count * noise_factor)
    return max(5, noisy_count)  # ensure at least 5 vehicles


def get_avg_speed(vehicle_count):
    """
    Compute average speed (km/h) inversely related to vehicle count.
    Fewer vehicles → higher speed; more vehicles → lower speed.
    """
    # Map vehicle count (roughly 5-500) to speed (roughly 10-80 km/h)
    max_vehicles = 500
    max_speed = 80.0
    min_speed = 10.0

    # Linear inverse relationship
    speed = max_speed - ((vehicle_count / max_vehicles) * (max_speed - min_speed))

    # Add ±5 km/h noise
    noise = numpy.random.uniform(-5.0, 5.0)
    speed = speed + noise

    # Clamp to realistic bounds
    speed = max(5.0, min(90.0, speed))
    return round(speed, 2)


def determine_congestion_level(vehicle_count, avg_speed):
    """
    Derive congestion level from vehicle count and average speed.
    Returns: 2 = High, 1 = Moderate, 0 = Low
    """
    if vehicle_count > 350 and avg_speed < 25:
        return 2  # High congestion
    elif vehicle_count > 200 or avg_speed < 40:
        return 1  # Moderate congestion
    else:
        return 0  # Low congestion


def generate_records(num_records):
    """Generate a list of traffic data records."""
    records = []

    for _ in range(num_records):
        # Randomly pick a road, day, and hour
        road = random.choice(ROADS)
        day = random.choice(DAYS)
        hour = random.randint(0, 23)

        # Generate features
        vehicle_count = get_vehicle_count_for_hour(hour)
        avg_speed = get_avg_speed(vehicle_count)
        congestion = determine_congestion_level(vehicle_count, avg_speed)

        records.append({
            'road_name': road,
            'day_of_week': day,
            'hour_of_day': hour,
            'vehicle_count': vehicle_count,
            'avg_speed_kmh': avg_speed,
            'congestion_level': congestion,
        })

    return records


def save_to_csv(records, filename):
    """Write records to a CSV file with the required header."""
    fieldnames = [
        'road_name', 'day_of_week', 'hour_of_day',
        'vehicle_count', 'avg_speed_kmh', 'congestion_level'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Dataset saved to '{filename}' with {len(records)} records.")


def print_class_summary(records):
    """Print counts for each congestion level and total records."""
    label_names = {0: 'Low', 1: 'Moderate', 2: 'High'}
    counts = {0: 0, 1: 0, 2: 0}

    for record in records:
        counts[record['congestion_level']] += 1

    print("\n--- Class Distribution ---")
    for level, name in label_names.items():
        print(f"  {name} (class {level}): {counts[level]} records")
    print(f"  Total: {len(records)} records")


# --- Main ---
if __name__ == '__main__':
    print(f"Generating {TOTAL_RECORDS} simulated traffic records...")

    records = generate_records(TOTAL_RECORDS)
    save_to_csv(records, OUTPUT_FILE)
    print_class_summary(records)
