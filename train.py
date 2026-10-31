"""
Used Car Price Predictor - Training Pipeline
Author: AI Pair Programmer (Antigravity)
Dataset: Indian Used Car Market (CarDekho)

This script:
1. Loads and cleans raw used car dataset (duplicates, missing values).
2. Engineers useful features like car_age and brand.
3. Builds an end-to-end ColumnTransformer & Pipeline with OneHotEncoder.
4. Trains and evaluates Linear Regression, Random Forest, and Gradient Boosting.
5. Displays a clear comparison table (R2, MAE, RMSE).
6. Generates 3 diagnostic charts:
   - model_comparison.png
   - actual_vs_predicted.png
   - feature_importance.png
7. Saves the best performing model pipeline to 'model.pkl' and metadata to 'metrics.json'.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300

# Reference year for age calculation
REFERENCE_YEAR = 2024


def load_and_preprocess_data(filepath="data.csv"):
    """
    Load dataset and perform data cleaning and feature engineering.
    """
    print(f"[1/5] Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    print(f"      Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Remove duplicate entries
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"      Removed {duplicates_count} duplicate rows. Remaining: {len(df)}")

    # 2. Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna().reset_index(drop=True)
        print(f"      Dropped rows with missing values. Remaining: {len(df)}")
    else:
        print("      No missing values found.")

    # 3. Feature engineering: Extract brand from car name
    # e.g., 'Maruti 800 AC' -> 'Maruti'
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0].strip().title())

    # 4. Feature engineering: Calculate car age
    # car_age = REFERENCE_YEAR - year (bounded to >= 0)
    df["car_age"] = REFERENCE_YEAR - df["year"]
    df["car_age"] = df["car_age"].apply(lambda age: max(0, int(age)))

    # Ensure selling_price and km_driven are positive numeric values
    df = df[df["selling_price"] > 0]
    df = df[df["km_driven"] > 0]

    print(f"      Unique brands found ({len(df['brand'].unique())}): {', '.join(sorted(df['brand'].unique())[:8])}...")
    print(f"      Cleaned dataset shape: {df.shape}")
    return df


def build_preprocessor(categorical_features, numerical_features):
    """
    Build a ColumnTransformer for preprocessing numerical and categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    return preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train Linear Regression, Random Forest, and Gradient Boosting Regressors.
    Evaluate each model on R2, MAE, and RMSE.
    """
    print("\n[2/5] Training and evaluating candidate models...")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
    }

    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        print(f"      Training {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # Compute RMSE across scikit-learn versions compatibility
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "y_pred": y_pred,
        }
        trained_pipelines[name] = pipeline

    # Display comparison table
    print("\n" + "=" * 65)
    print("                MODEL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"{'Model':<22} | {'R2 Score':<10} | {'MAE (INR)':<14} | {'RMSE (INR)':<14}")
    print("-" * 65)
    for name, metrics in results.items():
        print(
            f"{name:<22} | {metrics['R2']:<10.4f} | Rs. {metrics['MAE']:<9,.0f} | Rs. {metrics['RMSE']:<9,.0f}"
        )
    print("=" * 65)

    return results, trained_pipelines


def save_diagnostic_plots(results, trained_pipelines, best_model_name, X_test, y_test):
    """
    Generate and save the 3 required diagnostic visualizations:
    1. model_comparison.png
    2. actual_vs_predicted.png
    3. feature_importance.png
    """
    print("\n[3/5] Generating diagnostic plots...")

    # ---------------- 1. Model Comparison Chart ----------------
    models = list(results.keys())
    r2_vals = [results[m]["R2"] for m in models]
    mae_vals = [results[m]["MAE"] / 100000 for m in models]  # in Lakhs
    rmse_vals = [results[m]["RMSE"] / 100000 for m in models]  # in Lakhs

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # R2 plot
    bars1 = axes[0].bar(models, r2_vals, color=["#3498db", "#2ecc71", "#e67e22"], width=0.55)
    axes[0].set_title("R² Score (Higher is Better)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R²")
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontweight="bold")

    # MAE plot (in Lakhs)
    bars2 = axes[1].bar(models, mae_vals, color=["#3498db", "#2ecc71", "#e67e22"], width=0.55)
    axes[1].set_title("Mean Absolute Error (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_ylabel("MAE (₹ in Lakhs)")
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"₹{yval:.2f}L", ha="center", fontweight="bold")

    # RMSE plot (in Lakhs)
    bars3 = axes[2].bar(models, rmse_vals, color=["#3498db", "#2ecc71", "#e67e22"], width=0.55)
    axes[2].set_title("Root Mean Squared Error (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    axes[2].set_ylabel("RMSE (₹ in Lakhs)")
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"₹{yval:.2f}L", ha="center", fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Model Evaluation & Comparison Metrics", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("model_comparison.png", bbox_inches="tight")
    plt.close()
    print("      Saved 'model_comparison.png'")

    # ---------------- 2. Actual vs Predicted Chart ----------------
    best_preds = results[best_model_name]["y_pred"]
    y_test_np = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    # Convert to Lakhs for easier reading
    actual_lakhs = y_test_np / 100000
    pred_lakhs = best_preds / 100000

    plt.figure(figsize=(8, 7))
    plt.scatter(actual_lakhs, pred_lakhs, alpha=0.55, color="#1f77b4", edgecolors="none", s=40, label="Test Data Points")
    
    # Perfect prediction 45-degree line
    max_val = max(actual_lakhs.max(), pred_lakhs.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Ideal (Actual = Predicted)")

    plt.title(f"Actual vs Predicted Car Prices ({best_model_name})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Actual Selling Price (₹ Lakhs)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Selling Price (₹ Lakhs)", fontsize=11, fontweight="bold")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Annotate stats
    best_r2 = results[best_model_name]["R2"]
    best_mae = results[best_model_name]["MAE"]
    plt.annotate(
        f"Model: {best_model_name}\nR² = {best_r2:.3f}\nMAE = ₹{best_mae:,.0f}",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", bbox_inches="tight")
    plt.close()
    print("      Saved 'actual_vs_predicted.png'")

    # ---------------- 3. Feature Importance Chart ----------------
    # Extract feature importances if tree-based model
    best_pipe = trained_pipelines[best_model_name]
    regressor = best_pipe.named_steps["regressor"]
    preprocessor = best_pipe.named_steps["preprocessor"]

    if hasattr(regressor, "feature_importances_"):
        # Extract one-hot encoded and numerical feature names
        num_cols = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1]
        cat_cols = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        all_features = list(num_cols) + list(cat_cols)

        importances = regressor.feature_importances_
        feat_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title(f"Top 15 Feature Importances ({best_model_name})", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
        plt.ylabel("Feature", fontsize=11, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("feature_importance.png", bbox_inches="tight")
        plt.close()
        print("      Saved 'feature_importance.png'")
    else:
        # If best is Linear Regression, use coefficients
        num_cols = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1]
        cat_cols = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        all_features = list(num_cols) + list(cat_cols)
        coefs = np.abs(regressor.coef_)
        feat_df = pd.DataFrame({"Feature": all_features, "Importance": coefs})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="magma", legend=False)
        plt.title(f"Top 15 Feature Coefficients Magnitude ({best_model_name})", fontsize=14, fontweight="bold")
        plt.xlabel("Absolute Coefficient Magnitude", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig("feature_importance.png", bbox_inches="tight")
        plt.close()
        print("      Saved 'feature_importance.png'")


def main():
    # 1. Load and clean
    df = load_and_preprocess_data("data.csv")

    # Features and Target
    categorical_features = ["brand", "fuel", "seller_type", "transmission", "owner"]
    numerical_features = ["car_age", "km_driven"]
    all_features = numerical_features + categorical_features

    X = df[all_features]
    y = df["selling_price"]

    # Store lists of available choices for app UI
    available_brands = sorted(df["brand"].unique().tolist())
    available_fuels = sorted(df["fuel"].unique().tolist())
    available_sellers = sorted(df["seller_type"].unique().tolist())
    available_transmissions = sorted(df["transmission"].unique().tolist())
    available_owners = sorted(df["owner"].unique().tolist())

    # Split 80/20 train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # Build ColumnTransformer
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # Train and evaluate models
    results, trained_pipelines = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # Select best model based on R2 score
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n[4/5] Best performing model: {best_model_name}")
    print(f"      R2: {best_metrics['R2']:.4f}")
    print(f"      MAE: Rs. {best_metrics['MAE']:,.2f}")
    print(f"      RMSE: Rs. {best_metrics['RMSE']:,.2f}")

    # Generate diagnostic plots
    save_diagnostic_plots(results, trained_pipelines, best_model_name, X_test, y_test)

    # Attach metadata to best pipeline
    best_pipeline.best_model_name_ = best_model_name
    best_pipeline.mae_ = float(best_metrics["MAE"])
    best_pipeline.r2_ = float(best_metrics["R2"])
    best_pipeline.rmse_ = float(best_metrics["RMSE"])
    best_pipeline.reference_year_ = REFERENCE_YEAR
    best_pipeline.available_brands_ = available_brands
    best_pipeline.available_fuels_ = available_fuels
    best_pipeline.available_sellers_ = available_sellers
    best_pipeline.available_transmissions_ = available_transmissions
    best_pipeline.available_owners_ = available_owners

    # 4. Save best pipeline as model.pkl
    print("\n[5/5] Saving model and artifacts...")
    joblib.dump(best_pipeline, "model.pkl")
    print("      Successfully saved 'model.pkl'!")

    # Also save metadata json for easy inspection or lightweight app consumption
    metrics_summary = {
        "best_model": best_model_name,
        "reference_year": REFERENCE_YEAR,
        "metrics": {
            model: {
                "r2": round(results[model]["R2"], 4),
                "mae": round(results[model]["MAE"], 2),
                "rmse": round(results[model]["RMSE"], 2),
            }
            for model in results
        },
        "available_brands": available_brands,
        "available_fuels": available_fuels,
        "available_sellers": available_sellers,
        "available_transmissions": available_transmissions,
        "available_owners": available_owners,
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print("      Successfully saved 'metrics.json'!")

    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
