"""
Used Car Price Predictor & Deal Checker (Indian Market)
Author: AI Pair Programmer (Antigravity)
App Framework: Streamlit

Features:
- Car details input (Brand, Year, Km Driven, Fuel, Transmission, Owner Type, Asking Price)
- ML Model inference with Fair Valuation & Confidence Range (± MAE)
- Deal Verdict Badge: Good Deal, Fair Deal, or Overpriced
- Visual comparisons: Asking Price vs Predicted Fair Price
- Dynamic "What-If" sensitivity analysis (Age & Mileage curves)
- Feature Importance breakdown
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configure page settings
st.set_page_config(
    page_title="Used Car Price Predictor & Deal Checker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and badges
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
    }
    .good-deal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    .fair-deal {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    .overpriced-deal {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stat-title {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_metadata():
    """Load the trained model pipeline and accompanying metadata."""
    if not os.path.exists("model.pkl"):
        return None, None

    model = joblib.load("model.pkl")

    # Load metrics metadata if available
    metadata = {}
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return model, metadata


def format_inr(val):
    """Format numbers into Indian Rupee strings with Lakhs indication."""
    val_int = int(round(val))
    lakhs = val / 100000.0
    return f"₹{val_int:,.0f} (~{lakhs:.2f} Lakhs)"


def main():
    # Application Header
    st.markdown('<div class="main-header">🚗 Used Car Price Predictor & Deal Checker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Indian Pre-Owned Vehicle Fair Valuation & Intelligent Deal Assessment Engine</div>',
        unsafe_allow_html=True,
    )

    # Check for model existence
    model, metadata = load_model_and_metadata()

    if model is None:
        st.error(
            "⚠️ Model file `model.pkl` not found! Please run the training pipeline first: `python train.py`"
        )
        st.stop()

    # Extract metadata lists
    reference_year = getattr(model, "reference_year_", 2024)
    model_mae = getattr(model, "mae_", 80000.0)
    best_model_name = getattr(model, "best_model_name_", "Random Forest / Gradient Boosting")

    default_brands = getattr(
        model,
        "available_brands_",
        [
            "Maruti",
            "Hyundai",
            "Honda",
            "Tata",
            "Toyota",
            "Mahindra",
            "Ford",
            "Chevrolet",
            "Renault",
            "Volkswagen",
        ],
    )
    default_fuels = getattr(model, "available_fuels_", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    default_transmissions = getattr(model, "available_transmissions_", ["Manual", "Automatic"])
    default_owners = getattr(
        model,
        "available_owners_",
        ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
    )
    default_sellers = getattr(
        model,
        "available_sellers_",
        ["Individual", "Dealer", "Trustmark Dealer"],
    )

    # Sidebar / Input Panel
    st.sidebar.header("📋 Car & Deal Specifications")
    st.sidebar.caption("Provide the car specifications and seller's price:")

    # Car Brand
    brand = st.sidebar.selectbox("Brand / Make", options=default_brands, index=default_brands.index("Maruti") if "Maruti" in default_brands else 0)

    # Manufacturing Year
    current_year = 2024
    year = st.sidebar.slider(
        "Manufacturing Year",
        min_value=2000,
        max_value=current_year,
        value=2018,
        step=1,
    )

    # Km Driven
    km_driven = st.sidebar.number_input(
        "Kilometers Driven (km)",
        min_value=500,
        max_value=400000,
        value=48000,
        step=2500,
    )

    # Fuel Type
    fuel = st.sidebar.selectbox("Fuel Type", options=default_fuels, index=default_fuels.index("Petrol") if "Petrol" in default_fuels else 0)

    # Transmission
    transmission = st.sidebar.selectbox("Transmission", options=default_transmissions, index=0)

    # Ownership History
    owner = st.sidebar.selectbox("Ownership Status", options=default_owners, index=0)

    # Seller Type
    seller_type = st.sidebar.selectbox("Seller Type", options=default_sellers, index=0)

    st.sidebar.markdown("---")
    # Asking Price
    st.sidebar.subheader("💰 Seller Asking Price")
    asking_price = st.sidebar.number_input(
        "Asking Price (₹ INR)",
        min_value=10000,
        max_value=10000000,
        value=450000,
        step=10000,
    )

    # Prepare DataFrame for model input
    car_age = max(0, reference_year - year)
    input_data = pd.DataFrame(
        [
            {
                "car_age": car_age,
                "km_driven": km_driven,
                "brand": brand,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
            }
        ]
    )

    # Prediction
    predicted_price = float(model.predict(input_data)[0])
    # Lower bound cannot be negative
    lower_bound = max(15000.0, predicted_price - model_mae)
    upper_bound = predicted_price + model_mae

    # Valuation & Deal Analysis
    diff_val = asking_price - predicted_price
    diff_percent = (diff_val / predicted_price) * 100.0

    # Layout: Top Valuation Summary Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-title">Seller's Asking Price</div>
                <div class="stat-value" style="color: #0284c7;">₹{asking_price:,.0f}</div>
                <div style="font-size: 0.9rem; color: #64748b;">({asking_price/100000:.2f} Lakhs)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-title">Predicted Fair Price</div>
                <div class="stat-value" style="color: #10b981;">₹{predicted_price:,.0f}</div>
                <div style="font-size: 0.9rem; color: #64748b;">({predicted_price/100000:.2f} Lakhs)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-title">Estimated Fair Range (± MAE)</div>
                <div class="stat-value" style="font-size: 1.15rem; color: #475569; padding-top: 0.35rem;">
                    ₹{lower_bound:,.0f} – ₹{upper_bound:,.0f}
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">Tolerance band: ±₹{model_mae:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Deal Verdict Badge
    # Criteria:
    # Asking price > 10% below predicted -> Good Deal
    # Asking price > 10% above predicted -> Overpriced
    # Within 10% -> Fair Deal
    if asking_price < predicted_price * 0.90:
        verdict_class = "good-deal"
        verdict_title = "✅ GOOD DEAL"
        verdict_desc = f"The asking price is <b>{abs(diff_percent):.1f}% below</b> the estimated fair market price! You save approximately ₹{abs(diff_val):,.0f}."
    elif asking_price > predicted_price * 1.10:
        verdict_class = "overpriced-deal"
        verdict_title = "⚠️ OVERPRICED DEAL"
        verdict_desc = f"The asking price is <b>{diff_percent:.1f}% higher</b> than the estimated fair value (+₹{diff_val:,.0f}). Consider negotiating down to around ₹{predicted_price:,.0f}."
    else:
        verdict_class = "fair-deal"
        verdict_title = "⚖️ FAIR MARKET DEAL"
        verdict_desc = f"The asking price is within <b>±10%</b> of the expected fair price ({diff_percent:+.1f}%). It matches reasonable current market rates."

    st.markdown(
        f"""
        <div class="badge-card {verdict_class}">
            <h2 style="margin: 0; padding-bottom: 0.3rem;">{verdict_title}</h2>
            <div style="font-size: 1.05rem;">{verdict_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Comparison: Asking Price vs Predicted Price
    st.subheader("📊 Price Comparison Visualization")
    fig_comp, ax_comp = plt.subplots(figsize=(9, 2.8))

    categories = ["Seller Asking Price", "Predicted Fair Price"]
    values = [asking_price / 100000, predicted_price / 100000]
    bar_colors = ["#ef4444" if asking_price > predicted_price * 1.10 else ("#10b981" if asking_price < predicted_price * 0.90 else "#f59e0b"), "#2563eb"]

    bars = ax_comp.barh(categories, values, color=bar_colors, height=0.45)
    ax_comp.set_xlabel("Price in ₹ Lakhs", fontsize=10, fontweight="bold")
    ax_comp.set_xlim(0, max(values) * 1.35)

    # Draw shaded fair range band
    ax_comp.axvspan(lower_bound / 100000, upper_bound / 100000, color="#2563eb", alpha=0.15, label="Fair Price Band (± MAE)")

    # Data value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax_comp.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"₹{w:.2f} L", va="center", fontweight="bold", fontsize=10)

    ax_comp.legend(loc="lower right", frameon=True)
    ax_comp.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close()

    # "What-If" Sensitivity Analysis
    st.markdown("---")
    st.subheader("🔍 'What-If' Depreciation & Mileage Analysis")
    st.caption("See how the predicted car price adapts under different mileage and age scenarios:")

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        st.markdown("**Impact of Mileage (Kilometers Driven):**")
        km_test_values = [10000, 25000, 45000, 70000, 100000, 140000, 180000]
        km_df = pd.DataFrame(
            [
                {
                    "car_age": car_age,
                    "km_driven": km_val,
                    "brand": brand,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                }
                for km_val in km_test_values
            ]
        )
        km_preds = model.predict(km_df) / 100000.0  # In Lakhs

        fig_km, ax_km = plt.subplots(figsize=(6, 3.8))
        ax_km.plot([k / 1000 for k in km_test_values], km_preds, marker="o", color="#0284c7", linewidth=2.5, markersize=6)
        # Highlight current point
        curr_km_k = km_driven / 1000.0
        curr_pred_l = predicted_price / 100000.0
        ax_km.scatter([curr_km_k], [curr_pred_l], color="#dc2626", s=100, zorder=5, label="Current Selection")
        ax_km.set_xlabel("Kilometers Driven (in '000s km)", fontsize=9, fontweight="bold")
        ax_km.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=9, fontweight="bold")
        ax_km.set_title("Price vs Kilometers Driven", fontsize=11, fontweight="bold")
        ax_km.grid(True, linestyle="--", alpha=0.5)
        ax_km.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_km)
        plt.close()

    with col_sim2:
        st.markdown("**Impact of Vehicle Age (Years):**")
        age_test_values = list(range(1, 13))
        age_df = pd.DataFrame(
            [
                {
                    "car_age": age_val,
                    "km_driven": km_driven,
                    "brand": brand,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                }
                for age_val in age_test_values
            ]
        )
        age_preds = model.predict(age_df) / 100000.0  # In Lakhs

        fig_age, ax_age = plt.subplots(figsize=(6, 3.8))
        ax_age.plot(age_test_values, age_preds, marker="s", color="#7c3aed", linewidth=2.5, markersize=6)
        # Highlight current age point
        ax_age.scatter([car_age], [curr_pred_l], color="#dc2626", s=100, zorder=5, label="Current Selection")
        ax_age.set_xlabel("Vehicle Age (Years)", fontsize=9, fontweight="bold")
        ax_age.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=9, fontweight="bold")
        ax_age.set_title("Price vs Vehicle Age (Depreciation Curve)", fontsize=11, fontweight="bold")
        ax_age.grid(True, linestyle="--", alpha=0.5)
        ax_age.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_age)
        plt.close()

    # Feature Importance Breakdown
    st.markdown("---")
    st.subheader("🧠 What Factors Drive Car Prices Most?")
    st.caption("Feature importance derived from the trained machine learning pipeline:")

    # Attempt to extract feature importance from the model pipeline
    try:
        regressor = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]

        if hasattr(regressor, "feature_importances_"):
            num_cols = preprocessor.transformers_[0][2]
            cat_encoder = preprocessor.transformers_[1][1]
            cat_cols = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
            all_feat_names = list(num_cols) + list(cat_cols)

            fi_df = pd.DataFrame({"Feature": all_feat_names, "Importance": regressor.feature_importances_})
            fi_top = fi_df.sort_values(by="Importance", ascending=False).head(10)

            fig_fi, ax_fi = plt.subplots(figsize=(9, 3.5))
            ax_fi.barh(fi_top["Feature"][::-1], fi_top["Importance"][::-1], color="#0d9488", height=0.55)
            ax_fi.set_xlabel("Relative Importance Score", fontsize=9, fontweight="bold")
            ax_fi.set_title(f"Top 10 Most Influential Features ({best_model_name})", fontsize=11, fontweight="bold")
            ax_fi.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close()
        else:
            st.info("Feature importance plot is available when trained with Random Forest or Gradient Boosting.")
    except Exception as e:
        st.warning(f"Feature importance display note: {e}")

    # Footer and Methodology note
    st.markdown("---")
    st.caption(
        "📌 **Note**: Valuations are statistical predictions based on historical Indian used car sales data. Actual on-ground market prices may vary depending on vehicle mechanical condition, accident history, city registration, and buyer-seller negotiations."
    )


if __name__ == "__main__":
    main()
