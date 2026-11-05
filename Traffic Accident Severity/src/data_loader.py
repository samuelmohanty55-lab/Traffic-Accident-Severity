"""
Data loading and sampling utilities for Road Traffic Accident Severity Analysis.
Supports loading raw CSVs, stratified sampling, and realistic synthetic data generation.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
    SERIOUS_THRESHOLD,
)


def generate_representative_sample(
    n_samples: int = 50000,
    save_path: Optional[Path] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generates a statistically representative US Accidents sample mirroring the real dataset schema.
    Useful for immediate testing, pipeline verification, and rapid development.
    """
    np.random.seed(random_state)

    # 1. Temporal data across 2 years
    start_dates = pd.date_range("2021-01-01", "2022-12-31", periods=n_samples)
    noise_hours = np.random.randint(0, 24, size=n_samples)
    noise_minutes = np.random.randint(0, 60, size=n_samples)
    timestamps = [
        dt + pd.Timedelta(hours=int(h), minutes=int(m))
        for dt, h, m in zip(start_dates, noise_hours, noise_minutes)
    ]

    # 2. Weather conditions
    weather_types = [
        "Fair", "Clear", "Mostly Cloudy", "Partly Cloudy", "Overcast",
        "Light Rain", "Rain", "Heavy Rain", "Fog", "Light Snow", "Snow", "Thunderstorm"
    ]
    weather_probs = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.015, 0.015, 0.005, 0.003, 0.002]
    weather_conds = np.random.choice(weather_types, size=n_samples, p=weather_probs)

    # 3. Numeric variables with physical realism
    temp = np.random.normal(loc=61.0, scale=18.0, size=n_samples)
    temp = np.clip(temp, -10.0, 115.0)

    humidity = np.random.normal(loc=64.0, scale=22.0, size=n_samples)
    humidity = np.clip(humidity, 5.0, 100.0)

    pressure = np.random.normal(loc=29.5, scale=0.5, size=n_samples)
    pressure = np.clip(pressure, 26.0, 31.0)

    # Visibility: Mostly 10 miles, lower during fog/rain/snow
    vis = np.random.exponential(scale=3.0, size=n_samples) + 7.0
    vis = np.clip(vis, 0.0, 10.0)
    # Impair visibility under adverse weather
    adverse_mask = np.isin(weather_conds, ["Fog", "Heavy Rain", "Snow"])
    vis[adverse_mask] = np.random.uniform(0.1, 2.5, size=adverse_mask.sum())

    wind_speed = np.random.exponential(scale=6.5, size=n_samples)
    wind_speed = np.clip(wind_speed, 0.0, 50.0)

    precip = np.zeros(n_samples)
    rain_mask = np.isin(weather_conds, ["Light Rain", "Rain", "Heavy Rain", "Thunderstorm", "Light Snow", "Snow"])
    precip[rain_mask] = np.random.exponential(scale=0.08, size=rain_mask.sum())

    distance = np.random.exponential(scale=0.6, size=n_samples)
    distance = np.clip(distance, 0.0, 15.0)

    # 4. Road infrastructure flags (binary)
    junction = (np.random.rand(n_samples) < 0.14).astype(bool)
    traffic_signal = (np.random.rand(n_samples) < 0.18).astype(bool)
    crossing = (np.random.rand(n_samples) < 0.11).astype(bool)
    stop = (np.random.rand(n_samples) < 0.02).astype(bool)
    station = (np.random.rand(n_samples) < 0.03).astype(bool)
    railway = (np.random.rand(n_samples) < 0.01).astype(bool)
    amenity = (np.random.rand(n_samples) < 0.015).astype(bool)
    bump = (np.random.rand(n_samples) < 0.001).astype(bool)
    give_way = (np.random.rand(n_samples) < 0.004).astype(bool)
    no_exit = (np.random.rand(n_samples) < 0.001).astype(bool)
    roundabout = (np.random.rand(n_samples) < 0.001).astype(bool)
    traffic_calming = (np.random.rand(n_samples) < 0.001).astype(bool)

    side = np.random.choice(["R", "L"], size=n_samples, p=[0.82, 0.18])
    state = np.random.choice(["CA", "TX", "FL", "NY", "NC", "PA", "IL", "OH", "GA", "VA"], size=n_samples)

    # 5. Realistic Ground-Truth Severity Generation
    # Serious severity probability driven by: high speed/interstate (distance), night, fog/rain, junction, low visibility
    hours = np.array([ts.hour for ts in timestamps])
    is_night = (hours < 6) | (hours > 20)

    # Risk log-odds
    log_odds = -1.8
    log_odds += 0.8 * is_night.astype(float)
    log_odds += 0.9 * (vis < 2.0).astype(float)
    log_odds += 0.7 * junction.astype(float)
    log_odds += 0.6 * np.isin(weather_conds, ["Heavy Rain", "Fog", "Snow", "Thunderstorm"]).astype(float)
    log_odds += 0.4 * (wind_speed > 25).astype(float)
    log_odds += 0.15 * distance
    log_odds -= 0.5 * traffic_signal.astype(float)  # signals regulate speed
    # add noise
    log_odds += np.random.normal(0, 0.5, size=n_samples)

    prob_serious = 1 / (1 + np.exp(-log_odds))
    is_serious = (np.random.rand(n_samples) < prob_serious).astype(int)

    # Map to 1-4 severity scale for full fidelity
    severity = np.where(is_serious == 1,
                        np.random.choice([3, 4], size=n_samples, p=[0.75, 0.25]),
                        np.random.choice([1, 2], size=n_samples, p=[0.10, 0.90]))

    sunrise_sunset = np.where(is_night, "Night", "Day")

    df = pd.DataFrame({
        "ID": [f"A-{i+1000000}" for i in range(n_samples)],
        "Severity": severity,
        "Start_Time": timestamps,
        "Start_Lat": np.random.uniform(25.0, 48.0, size=n_samples),
        "Start_Lng": np.random.uniform(-124.0, -68.0, size=n_samples),
        "Distance(mi)": distance,
        "Side": side,
        "State": state,
        "Temperature(F)": temp,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": vis,
        "Wind_Speed(mph)": wind_speed,
        "Precipitation(in)": precip,
        "Weather_Condition": weather_conds,
        "Amenity": amenity,
        "Bump": bump,
        "Crossing": crossing,
        "Give_Way": give_way,
        "Junction": junction,
        "No_Exit": no_exit,
        "Railway": railway,
        "Roundabout": roundabout,
        "Station": station,
        "Stop": stop,
        "Traffic_Calming": traffic_calming,
        "Traffic_Signal": traffic_signal,
        "Sunrise_Sunset": sunrise_sunset,
    })

    # Introduce realistic missing values (5-10% in weather fields)
    for col in ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]:
        mask_nan = np.random.rand(n_samples) < 0.05
        df.loc[mask_nan, col] = np.nan

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DataLoader] Saved representative dataset with {n_samples} rows to {save_path}")

    return df


def load_dataset(file_path: Optional[Path] = None, sample_size: int = 50000) -> pd.DataFrame:
    """
    Loads dataset from raw directory if present.
    If not found, generates and caches a high-fidelity representative sample.
    """
    if file_path is None:
        csv_candidates = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_candidates:
            file_path = csv_candidates[0]
            print(f"[DataLoader] Found existing raw dataset: {file_path.name}")
        else:
            default_sample_path = RAW_DATA_DIR / "US_Accidents_Sample.csv"
            print(f"[DataLoader] No raw CSV found in {RAW_DATA_DIR}. Generating representative dataset...")
            return generate_representative_sample(n_samples=sample_size, save_path=default_sample_path)

    print(f"[DataLoader] Loading data from {file_path}...")
    # Check file size for memory-safe chunked sampling if > 500MB
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500 and sample_size:
        print(f"[DataLoader] Large file detected ({file_size_mb:.1f} MB). Sampling {sample_size} records...")
        chunks = []
        rows_so_far = 0
        chunk_size = 50000
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            rows_so_far += len(chunk)
            if rows_so_far >= sample_size:
                break
        df = pd.concat(chunks, ignore_index=True).iloc[:sample_size]
    else:
        df = pd.read_csv(file_path, low_memory=False)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    return df


if __name__ == "__main__":
    df = load_dataset(sample_size=20000)
    print(f"Dataset shape: {df.shape}")
    print(f"Severity distribution:\n{df['Severity'].value_counts(normalize=True)}")
