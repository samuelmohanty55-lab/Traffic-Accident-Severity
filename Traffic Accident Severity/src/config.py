"""
Configuration module for Road Traffic Accident Severity Prediction & Risk Analysis.
Defines paths, feature definitions, target mapping, and model parameters.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = BASE_DIR / "app"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, APP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Train / Validation / Test Splits (70 / 15 / 15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Target Definition (US Accidents: Severity 1 to 4)
# In Binary mode:
# Non-Serious (0): Severity 1 and 2 (Minor/Moderate delay & impact)
# Serious (1): Severity 3 and 4 (Significant / Severe traffic delay & major incident)
TARGET_COLUMN = "Severity"
BINARY_TARGET_COLUMN = "Is_Serious"
SERIOUS_THRESHOLD = 3  # Severity >= 3 is classified as Serious (1)

# Raw Feature Definitions for US Accidents
NUMERICAL_FEATURES = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Distance(mi)",
]

ROAD_FLAG_FEATURES = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
]

CATEGORICAL_FEATURES = [
    "Side",
    "Sunrise_Sunset",
    "Weather_Condition_Group",
    "Day_Part",
    "Season",
]

DERIVED_TEMPORAL_FEATURES = [
    "Hour",
    "DayOfWeek",
    "Month",
    "Is_Weekend",
    "Is_Rush_Hour",
    "Is_Night",
]

# Leakage columns to drop (features only known post-accident or identifiers)
LEAKAGE_AND_ID_COLUMNS = [
    "ID",
    "Source",
    "End_Time",
    "End_Lat",
    "End_Lng",
    "Description",
    "Street",
    "County",
    "Zipcode",
    "Country",
    "Timezone",
    "Airport_Code",
    "Weather_Timestamp",
    "Wind_Chill(F)",
    "Turning_Loop",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
]
