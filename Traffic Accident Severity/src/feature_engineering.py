"""
Feature engineering module for Accident Severity Prediction.
Extracts temporal, weather, and road risk attributes from raw US Accidents data.
"""

import pandas as pd
import numpy as np


def extract_temporal_features(df: pd.DataFrame, time_col: str = "Start_Time") -> pd.DataFrame:
    """
    Extracts time-of-day, rush hour, weekend, and seasonal indicators from timestamps.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Time components
    df["Hour"] = df[time_col].dt.hour.fillna(12).astype(int)
    df["DayOfWeek"] = df[time_col].dt.dayofweek.fillna(0).astype(int)
    df["Month"] = df[time_col].dt.month.fillna(1).astype(int)

    # Behavioral & environmental indicators
    df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    
    # Rush hours: 7:00-9:59 AM & 16:00-18:59 PM on weekdays
    is_morning_rush = (df["Hour"] >= 7) & (df["Hour"] <= 9)
    is_evening_rush = (df["Hour"] >= 16) & (df["Hour"] <= 18)
    df["Is_Rush_Hour"] = ((is_morning_rush | is_evening_rush) & (df["Is_Weekend"] == 0)).astype(int)

    # Night flag: 8 PM to 6 AM
    df["Is_Night"] = ((df["Hour"] >= 20) | (df["Hour"] < 6)).astype(int)

    # Part of day categorization
    def categorize_day_part(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    df["Day_Part"] = df["Hour"].apply(categorize_day_part)

    # Season mapping
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["Season"] = df["Month"].apply(get_season)

    return df


def categorize_weather(condition_str: str) -> str:
    """
    Groups hundreds of granular weather descriptions into consistent macro categories.
    """
    if pd.isna(condition_str):
        return "Unknown"
    cond = str(condition_str).lower()

    if any(k in cond for k in ["clear", "fair"]):
        return "Clear"
    elif any(k in cond for k in ["cloud", "overcast"]):
        return "Cloudy"
    elif any(k in cond for k in ["rain", "drizzle", "shower"]):
        return "Rain"
    elif any(k in cond for k in ["snow", "sleet", "ice", "blizzard", "freezing", "wintry"]):
        return "Snow/Ice"
    elif any(k in cond for k in ["fog", "mist", "haze", "smoke"]):
        return "Fog/Haze"
    elif any(k in cond for k in ["thunder", "t-storm", "storm", "funnel", "tornado", "squall"]):
        return "Severe_Storm"
    else:
        return "Other"


def engineer_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups weather conditions and creates physical threshold risk indicators.
    """
    df = df.copy()

    # Weather condition grouping
    if "Weather_Condition" in df.columns:
        df["Weather_Condition_Group"] = df["Weather_Condition"].apply(categorize_weather)
    else:
        df["Weather_Condition_Group"] = "Unknown"

    # Outlier / physical validity clipping
    if "Visibility(mi)" in df.columns:
        df["Visibility(mi)"] = df["Visibility(mi)"].clip(lower=0.0, upper=10.0)
        df["Low_Visibility"] = (df["Visibility(mi)"] < 2.0).astype(int)

    if "Temperature(F)" in df.columns:
        df["Temperature(F)"] = df["Temperature(F)"].clip(lower=-30.0, upper=130.0)
        df["Is_Freezing"] = (df["Temperature(F)"] <= 32.0).astype(int)

    if "Wind_Speed(mph)" in df.columns:
        df["Wind_Speed(mph)"] = df["Wind_Speed(mph)"].clip(lower=0.0, upper=80.0)
        df["High_Wind"] = (df["Wind_Speed(mph)"] > 25.0).astype(int)

    if "Precipitation(in)" in df.columns:
        df["Precipitation(in)"] = df["Precipitation(in)"].clip(lower=0.0, upper=10.0)
        df["Has_Precipitation"] = (df["Precipitation(in)"] > 0.0).astype(int)

    # Side standardisation
    if "Side" in df.columns:
        df["Side"] = df["Side"].fillna("R").apply(lambda s: s if s in ["R", "L"] else "R")

    # Sunrise/Sunset fallback
    if "Sunrise_Sunset" in df.columns:
        df["Sunrise_Sunset"] = df["Sunrise_Sunset"].fillna(
            df.get("Is_Night", pd.Series(0, index=df.index)).apply(lambda n: "Night" if n == 1 else "Day")
        )

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs full feature engineering pipeline on the raw accident dataframe.
    """
    df = extract_temporal_features(df)
    df = engineer_environmental_features(df)
    return df
