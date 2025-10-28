import pandas as pd
import joblib
import requests
import json
from datetime import datetime, timedelta, UTC
import warnings
import numpy as np
import os # To read API key from environment variable

# --- 1. Configuration ---
warnings.filterwarnings('ignore')

# --- Model & File Names ---
MODEL_HOURLY_FILE = 'model_hourly.pkl'
MODEL_DAILY_FILE = 'model_daily.pkl'
FORECAST_OUTPUT_FILE = 'forecasts.json' # Will be saved in the repo root

# --- API & Location Settings ---
LATITUDE = 13.0270199 # Peenya
LONGITUDE = 77.494094 # Peenya

# --- API URLs ---
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
OPENWEATHER_AQI_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# --- !!! GET API KEY FROM GITHUB SECRETS !!! ---
# GitHub Actions will inject the secret as an environment variable
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# --- Feature List (CRITICAL - Must match training) ---
FEATURES_LIST = [
    'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
    'calculated_aqi', 'temperature', 'humidity', 'wind_speed',
    'precipitation', 'pressure', 'hour', 'dayofweek', 'month',
    'dayofyear', 'weekofyear', 'aqi_lag_24hr', 'aqi_lag_48hr',
    'aqi_lag_168hr', 'aqi_roll_avg_24hr', 'aqi_roll_avg_168hr'
]

LONG_HISTORY_FEATURES = ['aqi_lag_168hr', 'aqi_roll_avg_168hr']
SHORT_HISTORY_FEATURES = ['aqi_lag_24hr', 'aqi_lag_48hr', 'aqi_roll_avg_24hr']


# --- 2. Helper Functions ---

def load_models():
    """Loads the trained models from disk."""
    print("Loading models...")
    try:
        models = {
            'hourly': joblib.load(MODEL_HOURLY_FILE),
            'daily': joblib.load(MODEL_DAILY_FILE)
        }
        print("Models loaded successfully.")
        return models
    except FileNotFoundError:
        print(f"Error: Model files not found. Make sure '{MODEL_HOURLY_FILE}' and '{MODEL_DAILY_FILE}' are in the repository root.")
        return None

def fetch_current_weather_and_aqi():
    """ Fetches current weather and current AQI pollutants."""
    print("Fetching current weather and AQI...")
    current_data = {}

    # --- Fetch Current Weather (using Open-Meteo 'current') ---
    try:
        weather_params = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m',
            'timezone': 'auto'
        }
        r_weather = requests.get(WEATHER_API_URL, params=weather_params)
        r_weather.raise_for_status()
        weather = r_weather.json()['current']
        current_data['temperature'] = weather.get('temperature_2m', 0)
        current_data['humidity'] = weather.get('relative_humidity_2m', 0)
        current_data['precipitation'] = weather.get('precipitation', 0)
        current_data['pressure'] = weather.get('pressure_msl', 0)
        current_data['wind_speed'] = weather.get('wind_speed_10m', 0)
        current_data['datetime'] = pd.to_datetime(weather.get('time'))
        print(f" -> Current weather fetched for {current_data['datetime']}.")
    except Exception as e:
        print(f"Error fetching current weather: {e}")
        return None

    # --- Fetch Current AQI (using OpenWeatherMap) ---
    if not OPENWEATHER_API_KEY:
        print("Error: OPENWEATHER_API_KEY environment variable not set.")
        # Attempt to proceed by filling pollutants with 0, but log the error clearly
        print("   -> Will proceed by filling pollutants with 0, but AQI calculation will be inaccurate.")
        for p in ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']:
             if p not in current_data: current_data[p] = 0
        # Ensure datetime exists even if weather fetch failed partially but key is missing
        if 'datetime' not in current_data: current_data['datetime'] = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Kolkata') # Fallback time
        return current_data


    try:
        aqi_params = {
            'lat': LATITUDE,
            'lon': LONGITUDE,
            'appid': OPENWEATHER_API_KEY
        }
        r_aqi = requests.get(OPENWEATHER_AQI_URL, params=aqi_params)
        r_aqi.raise_for_status()
        # Ensure 'list' exists and has at least one element
        aqi_list = r_aqi.json().get('list', [])
        if not aqi_list:
            raise ValueError("OpenWeatherMap API returned empty list for air pollution data.")
        pollutants = aqi_list[0].get('components', {})
        current_data['pm2_5'] = pollutants.get('pm2_5', 0)
        current_data['pm10'] = pollutants.get('pm10', 0)
        current_data['co'] = pollutants.get('co', 0) # OpenWeather CO is in µg/m³
        current_data['no2'] = pollutants.get('no2', 0)
        current_data['o3'] = pollutants.get('o3', 0)
        current_data['so2'] = pollutants.get('so2', 0)
        current_data['nh3'] = pollutants.get('nh3', 0)
        print(" -> Current AQI pollutants fetched.")
    except Exception as e:
        print(f"Error fetching current AQI: {e}")
        print("   -> Proceeding by filling pollutants with 0.")
        for p in ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']:
             if p not in current_data: current_data[p] = 0
         # Ensure datetime exists even if AQI fetch fails after successful weather fetch
        if 'datetime' not in current_data: current_data['datetime'] = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Kolkata') # Fallback time


    return current_data

def calculate_indian_aqi_subindex(Cp, pollutant):
    """Calculates the AQI sub-index for a single pollutant."""
    breakpoints = {
        'pm2_5': [((0, 30), (0, 50)), ((31, 60), (51, 100)), ((61, 90), (101, 200)), ((91, 120), (201, 300)), ((121, 250), (301, 400)), ((251, float('inf')), (401, 500))],
        'pm10': [((0, 50), (0, 50)), ((51, 100), (51, 100)), ((101, 250), (101, 200)), ((251, 350), (201, 300)), ((351, 430), (301, 400)), ((431, float('inf')), (401, 500))],
        'o3': [((0, 50), (0, 50)), ((51, 100), (51, 100)), ((101, 168), (101, 200)), ((169, 208), (201, 300)), ((209, 748), (301, 400)), ((749, float('inf')), (401, 500))],
        'co': [((0.0, 1.0), (0, 50)), ((1.1, 2.0), (51, 100)), ((2.1, 10.0), (101, 200)), ((10.1, 17.0), (201, 300)), ((17.1, 34.0), (301, 400)), ((34.1, float('inf')), (401, 500))], # Breakpoints in mg/m³
        'so2': [((0, 40), (0, 50)), ((41, 80), (51, 100)), ((81, 380), (101, 200)), ((381, 800), (201, 300)), ((801, 1600), (301, 400)), ((1601, float('inf')), (401, 500))],
        'no2': [((0, 40), (0, 50)), ((41, 80), (51, 100)), ((81, 180), (101, 200)), ((181, 280), (201, 300)), ((281, 400), (301, 400)), ((401, float('inf')), (401, 500))],
        'nh3': [((0, 200), (0, 50)), ((201, 400), (51, 100)), ((401, 800), (101, 200)), ((801, 1200), (201, 300)), ((1201, 1800), (301, 400)), ((1801, float('inf')), (401, 500))]
    }
    # OpenWeather provides CO in µg/m³. CPCB formula expects mg/m³ for lookup.
    Cp_lookup = Cp
    if pollutant == 'co': Cp_lookup = Cp / 1000.0
    if pd.isna(Cp_lookup) or Cp_lookup < 0: Cp_lookup = 0
    for (BP_LO, BP_HI), (I_LO, I_HI) in breakpoints.get(pollutant, []):
        # Use epsilon for float comparison with CO breakpoints
        if pollutant == 'co':
            if (BP_LO - 1e-9) <= Cp_lookup <= (BP_HI + 1e-9): return ((I_HI - I_LO) / (BP_HI - BP_LO)) * (Cp_lookup - BP_LO) + I_LO
        else:
             if BP_LO <= Cp_lookup <= BP_HI: return ((I_HI - I_LO) / (BP_HI - BP_LO)) * (Cp_lookup - BP_LO) + I_LO
    if Cp_lookup > 0: return 500 # Cap if above highest breakpoint
    return 0

def create_feature_vector(current_data):
    """ Creates the feature vector, filling missing historical features. """
    print("Engineering features from current data...")
    if current_data is None or 'datetime' not in current_data: return None
    pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']
    sub_indices = {p: calculate_indian_aqi_subindex(current_data.get(p, 0), p) for p in pollutants}
    current_aqi = max(sub_indices.values()) if sub_indices else 0
    current_data['calculated_aqi'] = current_aqi
    print(f" -> Calculated current AQI: {current_aqi:.2f}")
    dt = current_data['datetime']
    # Ensure datetime object has time attributes
    if isinstance(dt, pd.Timestamp):
        current_data['hour'] = dt.hour
        current_data['dayofweek'] = dt.dayofweek
        current_data['month'] = dt.month
        current_data['dayofyear'] = dt.dayofyear
        # Use dt.isocalendar().week which returns Int64
        weekofyear = dt.isocalendar().week
        current_data['weekofyear'] = int(weekofyear) if pd.notna(weekofyear) else 0 # Convert to int, handle NA
    else: # Fallback if dt is not a timestamp
         print("Warning: Could not extract time features from datetime.")
         current_data['hour'] = 0
         current_data['dayofweek'] = 0
         current_data['month'] = 1
         current_data['dayofyear'] = 1
         current_data['weekofyear'] = 1


    print(" -> Filling missing historical features...")
    # Use current AQI as the fill value for missing history
    fill_value = current_aqi
    for feat in SHORT_HISTORY_FEATURES + LONG_HISTORY_FEATURES:
        current_data[feat] = fill_value

    # Create DataFrame with the correct order and types
    feature_vector = pd.DataFrame(columns=FEATURES_LIST)
    for col in FEATURES_LIST:
        feature_vector.loc[0, col] = current_data.get(col, 0) # Use get with default 0

    # Ensure all columns are numeric, coercing errors to NaN then filling
    for col in feature_vector.columns:
        feature_vector[col] = pd.to_numeric(feature_vector[col], errors='coerce')

    feature_vector = feature_vector.fillna(0).astype(float) # Fill potential NaNs from coercion

    print("Feature vector created.")
    # Ensure final output matches FEATURES_LIST exactly
    return feature_vector[FEATURES_LIST]


def format_predictions(hourly_preds, daily_preds, current_data):
    """Formats the raw model outputs into a clean JSON."""
    hourly_values = hourly_preds[0]
    daily_values = daily_preds[0]
    now_utc = datetime.now(UTC) # Use UTC for generation timestamp consistency
    current_ts = current_data.get('datetime', pd.Timestamp.now(tz='UTC')) # Use data timestamp or fallback

    hourly_targets = [1, 2, 3, 6, 12, 18, 24]
    hourly_forecast = [{'time': (current_ts + timedelta(hours=h)).isoformat(), 'hours_ahead': h, 'aqi': round(hourly_values[i], 2)} for i, h in enumerate(hourly_targets)]
    daily_targets = [1, 2, 3, 4, 5, 6, 7]
    daily_forecast = []
    # Ensure current_ts is a Timestamp object to access .date()
    current_date = current_ts.date() if isinstance(current_ts, pd.Timestamp) else datetime.now(UTC).date()
    for i, d in enumerate(daily_targets):
         forecast_date = current_date + timedelta(days=d)
         daily_forecast.append({'date': forecast_date.strftime('%Y-%m-%d'), 'days_ahead': d, 'avg_aqi': round(daily_values[i], 2)})


    return {'forecast_generated_at_utc': now_utc.isoformat(), 'current_conditions_time': current_ts.isoformat(), 'current_aqi': round(current_data.get('calculated_aqi', 0), 2), 'location_coords': {'latitude': LATITUDE, 'longitude': LONGITUDE}, 'hourly_forecast': hourly_forecast, 'daily_forecast': daily_forecast}

# --- 3. Main Execution ---
def main():
    """Main function to run the complete forecast pipeline."""
    run_start_time = datetime.now()
    print(f"--- Forecast run started at {run_start_time} ---")
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
        print("!!! FATAL ERROR: OPENWEATHER_API_KEY environment variable not set or is placeholder. Set it in GitHub Secrets. !!!")
        # Exit with error code to make GitHub Actions fail
        exit(1)

    try:
        models = load_models()
        if models is None: exit(1) # Exit if models can't load

        current_data = fetch_current_weather_and_aqi()
        if current_data is None: exit(1) # Exit if data fetching fails critically

        feature_vector = create_feature_vector(current_data)
        if feature_vector is None: exit(1) # Exit if features can't be created

        # --- Check for NaNs before prediction ---
        if feature_vector.isnull().values.any():
            print("!!! FATAL ERROR: NaNs found in feature vector before prediction.")
            print(feature_vector[feature_vector.isnull().any(axis=1)])
            exit(1)


        print("Generating predictions...")
        hourly_preds = models['hourly'].predict(feature_vector)
        daily_preds = models['daily'].predict(feature_vector)
        output_json = format_predictions(hourly_preds, daily_preds, current_data)

        print(f"Saving predictions to '{FORECAST_OUTPUT_FILE}'...")
        with open(FORECAST_OUTPUT_FILE, 'w') as f: json.dump(output_json, f, indent=2)

        run_end_time = datetime.now()
        print(f"\n--- Forecast run COMPLETE (Duration: {run_end_time - run_start_time}) ---")
        print(f"Predictions successfully saved to '{FORECAST_OUTPUT_FILE}'")

    except requests.exceptions.RequestException as e:
        print(f"\n--- API ERROR ---")
        print(f"Failed to fetch data: {e}")
        exit(1) # Exit with error code
    except FileNotFoundError as e:
         print(f"\n--- FILE ERROR ---")
         print(f"Model file not found: {e}")
         exit(1)
    except Exception as e:
        print(f"\n--- SCRIPT ERROR ---")
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1) # Exit with error code

if __name__ == "__main__":
    main()

