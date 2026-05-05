
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "movies_cleaned.csv"
RESULTS_PATH = "stage4_ml_results_summary.txt"
FIGURES_DIR = "stage4_figures"

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

required_columns = ["runtime", "imdb_rating", "genre", "release_year"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns are: {list(df.columns)}"
    )

df = df.dropna(subset=required_columns).copy()

genre_counts = df["genre"].value_counts()
valid_genres = genre_counts[genre_counts >= 20].index
df = df[df["genre"].isin(valid_genres)].copy()

print(f"Dataset size after filtering: {len(df)} movies")
print(f"Number of genres used: {df['genre'].nunique()}")

X = df[["runtime", "release_year", "genre"]]
y = df["imdb_rating"]

numeric_features = ["runtime", "release_year"]
categorical_features = ["genre"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

models = {
    "Baseline Mean": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=12
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []
predictions = {}

for name, model in models.items():
    print(f"Training {name}...")

    if name == "Baseline Mean":
        pipeline = Pipeline([
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    predictions[name] = y_pred

results_df = pd.DataFrame(results).sort_values(by="MAE")
print("\nModel results:")
print(results_df)

plt.figure(figsize=(10, 5))
plt.bar(results_df["model"], results_df["MAE"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("MAE")
plt.title("Model Comparison by MAE")
plt.tight_layout()
plt.savefig(Path(FIGURES_DIR) / "model_comparison_mae.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(results_df["model"], results_df["RMSE"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("RMSE")
plt.title("Model Comparison by RMSE")
plt.tight_layout()
plt.savefig(Path(FIGURES_DIR) / "model_comparison_rmse.png", dpi=200)
plt.close()

best_model_name = results_df.iloc[0]["model"]
best_pred = predictions[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.5)
plt.xlabel("Actual IMDb Rating")
plt.ylabel("Predicted IMDb Rating")
plt.title(f"Actual vs Predicted Ratings ({best_model_name})")

min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])
plt.tight_layout()
plt.savefig(Path(FIGURES_DIR) / "actual_vs_predicted_best_model.png", dpi=200)
plt.close()

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=12
    ))
])
rf_pipeline.fit(X_train, y_train)

feature_names = numeric_features.copy()
encoded_genres = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
feature_names.extend(encoded_genres)

importances = rf_pipeline.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
plt.xlabel("Importance")
plt.title("Top Feature Importances from Random Forest")
plt.tight_layout()
plt.savefig(Path(FIGURES_DIR) / "random_forest_feature_importance.png", dpi=200)
plt.close()

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("STAGE 4 ML RESULTS SUMMARY\n")
    f.write("==========================\n\n")

    f.write("Research Question:\n")
    f.write("Can IMDb ratings be predicted using movie runtime, genre, and release year?\n\n")

    f.write("Dataset:\n")
    f.write(f"The cleaned dataset contains {len(df)} movies after filtering small genre groups.\n")
    f.write("The features used were runtime, release year, and genre.\n")
    f.write("The target variable was IMDb rating.\n\n")

    f.write("Models Tested:\n")
    for model_name in models.keys():
        f.write(f"- {model_name}\n")

    f.write("\nModel Performance:\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation:\n")
    f.write(
        "The models were compared using MAE, RMSE, and R2 score. "
        "The baseline model predicts the average IMDb rating, so useful ML models should perform better than this baseline. "
        "If tree-based models such as Random Forest or Gradient Boosting perform better than Linear Regression, this suggests that the relationship between runtime, genre, release year, and IMDb rating is not fully linear. "
        "However, IMDb ratings are subjective and depend on many factors that are not included in this dataset, such as acting quality, budget, director, audience expectations, and number of votes. "
        "Therefore, the model results should be interpreted as an approximate prediction rather than a perfect explanation of movie ratings.\n\n"
    )

    f.write("Main Conclusion:\n")
    f.write(
        "Runtime, genre, and release year provide some useful information for predicting IMDb rating, "
        "but these variables alone are not enough to fully explain audience ratings. "
        "This supports the EDA findings: movie length has a relationship with IMDb rating, but other factors also matter.\n"
    )

print("\nDone.")
print(f"Results saved to: {RESULTS_PATH}")
print(f"Figures saved to: {FIGURES_DIR}")
print(f"Best model based on MAE: {best_model_name}")
