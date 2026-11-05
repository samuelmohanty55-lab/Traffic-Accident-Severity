"""
Data Preprocessing and Pipeline Construction.
Builds scikit-learn ColumnTransformer and handles train/val/test stratified splitting.
Prevents data leakage by fitting transformers strictly on the training partition.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SERIOUS_THRESHOLD,
    TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
    NUMERICAL_FEATURES,
    ROAD_FLAG_FEATURES,
    CATEGORICAL_FEATURES,
    DERIVED_TEMPORAL_FEATURES,
    LEAKAGE_AND_ID_COLUMNS,
)
from src.feature_engineering import engineer_all_features


def prepare_raw_data(df: pd.DataFrame, drop_threshold: float = 0.50) -> pd.DataFrame:
    """
    Cleans raw data, drops high-missing columns & leakage columns, and creates binary target.
    """
    df = df.copy()

    # Create binary target: 1 = Serious (Severity 3, 4), 0 = Non-Serious (Severity 1, 2)
    if TARGET_COLUMN in df.columns:
        df[BINARY_TARGET_COLUMN] = (df[TARGET_COLUMN] >= SERIOUS_THRESHOLD).astype(int)

    # Apply feature engineering
    df = engineer_all_features(df)

    # Convert all boolean columns to integer (0/1) for scikit-learn SimpleImputer compatibility
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop leakage columns and identifiers if present
    cols_to_drop = [c for c in LEAKAGE_AND_ID_COLUMNS if c in df.columns]
    
    # Drop columns exceeding missing threshold
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    high_missing_cols = [c for c in high_missing_cols if c not in [TARGET_COLUMN, BINARY_TARGET_COLUMN]]

    all_drop = list(set(cols_to_drop + high_missing_cols))
    df = df.drop(columns=all_drop, errors="ignore")

    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identifies available numerical, road flag, and categorical columns in the dataframe.
    """
    num_cols = [c for c in NUMERICAL_FEATURES + DERIVED_TEMPORAL_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    road_cols = [c for c in ROAD_FLAG_FEATURES if c in df.columns]

    # Convert road flag booleans to int 0/1
    for col in road_cols:
        df[col] = df[col].fillna(False).astype(int)

    return num_cols, cat_cols, road_cols


def build_preprocessor_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    road_flag_cols: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer to handle imputation, scaling, and one-hot encoding.
    """
    # Numerical pipeline: Median imputation + optional standard scaling
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical pipeline: Most frequent imputation + OneHotEncoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Road flag pipeline: Impute missing with 0
    road_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    transformers = []
    if numerical_cols:
        transformers.append(("num", numeric_transformer, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    if road_flag_cols:
        transformers.append(("road", road_transformer, road_flag_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names output by the preprocessor.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if fitted before get_feature_names_out support
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat" and hasattr(trans.named_steps["encoder"], "get_feature_names_out"):
                names.extend(trans.named_steps["encoder"].get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names


def stratified_split(
    df: pd.DataFrame,
    target_col: str = BINARY_TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits data into Train (70%), Validation (15%), and Test (15%) sets using stratification.
    """
    y = df[target_col]
    X = df.drop(columns=[col for col in [TARGET_COLUMN, BINARY_TARGET_COLUMN] if col in df.columns])

    # First split: Train (70%) vs Temp (30%)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, stratify=y, random_state=random_state
    )

    # Second split: Validation (15%) vs Test (15%) -> equal split of temp (0.50)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    print(f"[Split] Total: {len(df)} records")
    print(f"[Split] Train set: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%), Serious class: {y_train.mean()*100:.2f}%")
    print(f"[Split] Val set:   {len(X_val)} ({len(X_val)/len(df)*100:.1f}%), Serious class: {y_val.mean()*100:.2f}%")
    print(f"[Split] Test set:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%), Serious class: {y_test.mean()*100:.2f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test
