
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer

TARGET = "y"
DROP_FROM_FEATURES = ["y", "duration"]

NUMERIC_COLUMNS = [
    "age", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    "contact_history", "prior_success"
]

CATEGORICAL_COLUMNS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
    "campaign_group", "age_group", "contact_month"
]

REQUIRED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y"
]

def clean_data(data):
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"В CSV отсутствуют обязательные столбцы: {missing}")
    result = data.copy()
    result = result.drop_duplicates().reset_index(drop=True)
    result["y"] = result["y"].map({"no": 0, "yes": 1})
    if result["y"].isna().any():
        raise ValueError("В целевой переменной y обнаружены неизвестные значения")
    return result

def add_features(data):
    result = data.copy()
    result["contact_history"] = (result["previous"] > 0).astype("int8")
    result["campaign_group"] = pd.cut(
        result["campaign"],
        bins=[0, 1, 3, 6, float("inf")],
        labels=["1", "2-3", "4-6", "7+"],
        include_lowest=True
    ).astype("object")
    result["prior_success"] = (result["poutcome"] == "success").astype("int8")
    result["age_group"] = pd.cut(
        result["age"],
        bins=[0, 29, 44, 59, float("inf")],
        labels=["<30", "30-44", "45-59", "60+"],
        include_lowest=True
    ).astype("object")
    result["contact_month"] = (
        result["contact"].astype(str) + "_" + result["month"].astype(str)
    )
    return result

def identity(data):
    return data.copy()

def make_pipeline(model, use_engineered=True):
    feature_step = FunctionTransformer(
        add_features if use_engineered else identity,
        validate=False
    )
    numeric_columns = NUMERIC_COLUMNS if use_engineered else [
        "age", "campaign", "pdays", "previous", "emp.var.rate",
        "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"
    ]
    categorical_columns = CATEGORICAL_COLUMNS if use_engineered else [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome"
    ]
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, numeric_columns),
        ("categorical", categorical, categorical_columns)
    ], remainder="drop")
    return Pipeline([
        ("features", feature_step),
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def prepare_xy(data):
    cleaned = clean_data(data)
    y = cleaned[TARGET].astype("int8")
    X = cleaned.drop(columns=DROP_FROM_FEATURES)
    return X, y


def prepare_file(path):
    data = pd.read_csv(path, sep=";")
    cleaned = clean_data(data)
    y = cleaned[TARGET].astype("int8")
    X = cleaned.drop(columns=DROP_FROM_FEATURES)
    return X, y
