"""
Amazon Sales Prediction — Streamlit App
========================================
Backend  : scikit-learn (Random Forest + Linear Regression)
Frontend : Streamlit + Plotly + Seaborn
Target   : rating_count  (proxy for sales / popularity volume)
Features : discounted_price, actual_price, discount_percentage,
           rating, main_category (label-encoded)
"""

import re
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Sales Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 700;
            color: #FF9900;
            text-align: center;
            padding: 1rem 0 0.3rem 0;
        }
        .sub-header {
            font-size: 1rem;
            color: #555;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #f7f8fa;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
        }
        .prediction-box {
            background: linear-gradient(135deg, #FF9900 0%, #e68a00 100%);
            color: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def clean_price(val: str) -> float:
    """Strip ₹ and commas, return float."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace("₹", "").replace(",", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def clean_percent(val: str) -> float:
    """Strip % and return float."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace("%", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def clean_count(val: str) -> float:
    """Remove commas from rating_count."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace(",", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def extract_main_category(cat: str) -> str:
    """Take the first segment of the pipe-separated category string."""
    if pd.isna(cat):
        return "Unknown"
    return str(cat).split("|")[0].strip()


# ─────────────────────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & preprocessing dataset …")
def load_data(filepath: str = "amazon.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath, on_bad_lines="skip")

    # Clean numeric columns
    df["discounted_price"] = df["discounted_price"].apply(clean_price)
    df["actual_price"] = df["actual_price"].apply(clean_price)
    df["discount_percentage"] = df["discount_percentage"].apply(clean_percent)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = df["rating_count"].apply(clean_count)

    # Extract top-level category
    df["main_category"] = df["category"].apply(extract_main_category)

    # Drop rows with NaN in critical columns
    required = [
        "discounted_price", "actual_price", "discount_percentage",
        "rating", "rating_count", "main_category",
    ]
    df.dropna(subset=required, inplace=True)
    df = df[df["rating_count"] > 0].reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING & MODEL TRAINING
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training prediction models …")
def train_models(df: pd.DataFrame):
    le = LabelEncoder()
    df = df.copy()
    df["category_enc"] = le.fit_transform(df["main_category"])

    feature_cols = [
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "category_enc",
    ]
    X = df[feature_cols].values
    y = np.log1p(df["rating_count"].values)   # log transform for better fit

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        # CV score
        cv = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
        results[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
            "CV R² (mean)": round(cv.mean(), 4),
            "CV R² (std)": round(cv.std(), 4),
        }
        trained[name] = model

    return trained, results, scaler, le, X_test, y_test, feature_cols


# ─────────────────────────────────────────────────────────────
#  APP LAYOUT
# ─────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-header">📦 Amazon Sales Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Predict product sales volume (rating count proxy) '
        "using Machine Learning on Amazon product data</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load Data
    df = load_data("amazon.csv")
    trained_models, model_results, scaler, le, X_test, y_test, feature_cols = train_models(df)

    # ── Sidebar
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
            width=140,
        )
        st.markdown("### ⚙️ Settings")
        selected_model = st.selectbox(
            "Choose Prediction Model",
            list(trained_models.keys()),
            index=0,
        )
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        st.metric("Total Products", len(df))
        st.metric("Categories", df["main_category"].nunique())
        st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
        st.metric("Avg Discount", f"{df['discount_percentage'].mean():.1f}%")
        st.markdown("---")
        st.markdown("**Made with ❤️ using Streamlit**")

    # ── Tabs
    tabs = st.tabs(
        [
            "🏠 Overview",
            "📈 EDA & Insights",
            "🤖 Model Performance",
            "🔮 Predict Sales",
            "📋 Data Table",
        ]
    )

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Project Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>📦</h3><b>Total Products</b><br>'
                        f'<span style="font-size:1.6rem;color:#FF9900">{len(df):,}</span></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>🏷️</h3><b>Categories</b><br>'
                        f'<span style="font-size:1.6rem;color:#FF9900">{df["main_category"].nunique()}</span></div>',
                        unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>⭐</h3><b>Avg Rating</b><br>'
                        f'<span style="font-size:1.6rem;color:#FF9900">{df["rating"].mean():.2f}</span></div>',
                        unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>💸</h3><b>Avg Discount</b><br>'
                        f'<span style="font-size:1.6rem;color:#FF9900">{df["discount_percentage"].mean():.1f}%</span></div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔍 How It Works")
        st.markdown(
            """
            | Step | Description |
            |------|-------------|
            | **1. Data Ingestion** | Load `amazon.csv`, parse ₹ prices, clean percentages & counts |
            | **2. Feature Engineering** | Extract `main_category`, encode labels, log-transform target |
            | **3. Model Training** | Train Random Forest, Gradient Boosting, Ridge & Linear Regression |
            | **4. Evaluation** | Compare MAE, RMSE, R², and 5-fold CV R² scores |
            | **5. Prediction** | Use the best model to forecast `rating_count` for new inputs |
            """
        )

        st.subheader("📌 Target Variable")
        st.info(
            "**`rating_count`** is used as a proxy for **sales volume**. "
            "A higher number of ratings correlates strongly with higher purchase volume, "
            "making it a reliable sales popularity indicator."
        )

    # ══════════════════════════════════════════════════════════
    #  TAB 2 — EDA
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Exploratory Data Analysis")

        # -- Distribution of rating_count (log scale)
        col_a, col_b = st.columns(2)
        with col_a:
            fig1 = px.histogram(
                df, x="rating_count", nbins=60,
                title="Distribution of Rating Count (Sales Volume Proxy)",
                color_discrete_sequence=["#FF9900"],
                log_x=True,
            )
            fig1.update_layout(xaxis_title="Rating Count (log scale)", yaxis_title="Frequency")
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            fig2 = px.histogram(
                df, x="rating", nbins=20,
                title="Distribution of Product Ratings",
                color_discrete_sequence=["#146EB4"],
            )
            fig2.update_layout(xaxis_title="Rating", yaxis_title="Frequency")
            st.plotly_chart(fig2, use_container_width=True)

        # -- Top categories by average rating_count
        cat_avg = (
            df.groupby("main_category")["rating_count"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .head(15)
        )
        fig3 = px.bar(
            cat_avg, x="main_category", y="rating_count",
            title="Top 15 Categories by Average Rating Count",
            color="rating_count",
            color_continuous_scale="Oranges",
            labels={"main_category": "Category", "rating_count": "Avg Rating Count"},
        )
        fig3.update_xaxes(tickangle=30)
        st.plotly_chart(fig3, use_container_width=True)

        # -- Scatter: discount vs rating_count
        col_c, col_d = st.columns(2)
        with col_c:
            fig4 = px.scatter(
                df.sample(min(500, len(df)), random_state=42),
                x="discount_percentage", y="rating_count",
                color="main_category",
                title="Discount % vs Rating Count",
                log_y=True,
                opacity=0.7,
                labels={"discount_percentage": "Discount %", "rating_count": "Rating Count"},
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col_d:
            fig5 = px.scatter(
                df.sample(min(500, len(df)), random_state=42),
                x="rating", y="rating_count",
                color="main_category",
                title="Rating vs Rating Count",
                log_y=True,
                opacity=0.7,
            )
            st.plotly_chart(fig5, use_container_width=True)

        # -- Correlation heatmap
        corr_cols = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
        corr = df[corr_cols].corr()
        fig6 = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig6.update_layout(title="Feature Correlation Heatmap")
        st.plotly_chart(fig6, use_container_width=True)

        # -- Price distribution by category (top 8)
        top_cats = df["main_category"].value_counts().head(8).index.tolist()
        fig7 = px.box(
            df[df["main_category"].isin(top_cats)],
            x="main_category", y="discounted_price",
            title="Discounted Price Distribution by Top 8 Categories",
            color="main_category",
            log_y=True,
        )
        fig7.update_xaxes(tickangle=30)
        st.plotly_chart(fig7, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Model Evaluation")

        # Metrics table
        metrics_df = pd.DataFrame(model_results).T.reset_index()
        metrics_df.rename(columns={"index": "Model"}, inplace=True)
        st.dataframe(
            metrics_df.style.highlight_max(subset=["R²", "CV R² (mean)"], color="#d4edda")
                            .highlight_min(subset=["MAE", "RMSE"], color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )

        col_e, col_f = st.columns(2)
        with col_e:
            fig8 = px.bar(
                metrics_df, x="Model", y="R²",
                title="R² Score Comparison",
                color="R²",
                color_continuous_scale="Blues",
                text_auto=".3f",
            )
            st.plotly_chart(fig8, use_container_width=True)

        with col_f:
            fig9 = px.bar(
                metrics_df, x="Model", y="RMSE",
                title="RMSE Comparison (lower is better)",
                color="RMSE",
                color_continuous_scale="Reds_r",
                text_auto=".3f",
            )
            st.plotly_chart(fig9, use_container_width=True)

        # Actual vs Predicted (selected model)
        model = trained_models[selected_model]
        preds_log = model.predict(X_test)
        actual_vals = np.expm1(y_test)
        pred_vals = np.expm1(preds_log)

        scatter_df = pd.DataFrame({"Actual": actual_vals, "Predicted": pred_vals})
        fig10 = px.scatter(
            scatter_df.sample(min(300, len(scatter_df)), random_state=1),
            x="Actual", y="Predicted",
            title=f"Actual vs Predicted — {selected_model}",
            log_x=True, log_y=True,
            opacity=0.6,
            color_discrete_sequence=["#FF9900"],
        )
        fig10.add_shape(
            type="line",
            x0=scatter_df["Actual"].min(), y0=scatter_df["Actual"].min(),
            x1=scatter_df["Actual"].max(), y1=scatter_df["Actual"].max(),
            line=dict(color="red", dash="dash"),
        )
        st.plotly_chart(fig10, use_container_width=True)

        # Feature importance (RF / GB only)
        if selected_model in ("Random Forest", "Gradient Boosting"):
            imp = model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": imp,
            }).sort_values("Importance", ascending=False)
            fig11 = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                title=f"Feature Importance — {selected_model}",
                color="Importance",
                color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig11, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 4 — PREDICT
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("🔮 Predict Sales Volume for a New Product")

        col1, col2 = st.columns(2)
        with col1:
            category_options = sorted(df["main_category"].unique().tolist())
            sel_cat = st.selectbox("Product Category", category_options)
            disc_price = st.number_input(
                "Discounted Price (₹)", min_value=1.0,
                max_value=500000.0, value=499.0, step=10.0
            )
            actual_price_val = st.number_input(
                "Actual / MRP Price (₹)", min_value=1.0,
                max_value=500000.0, value=999.0, step=10.0
            )

        with col2:
            disc_pct = st.slider("Discount Percentage (%)", 0, 95, 50)
            rating_val = st.slider("Product Rating (1.0 – 5.0)", 1.0, 5.0, 4.0, 0.1)
            st.markdown("---")
            model_choice = st.selectbox(
                "Model to use for prediction", list(trained_models.keys()), index=0
            )

        if st.button("🚀 Predict Sales Volume", type="primary", use_container_width=True):
            try:
                cat_enc = le.transform([sel_cat])[0]
            except ValueError:
                cat_enc = 0

            input_arr = np.array([[
                disc_price, actual_price_val, disc_pct, rating_val, cat_enc
            ]])
            input_scaled = scaler.transform(input_arr)
            log_pred = trained_models[model_choice].predict(input_scaled)[0]
            pred_count = int(np.expm1(log_pred))

            st.markdown(
                f'<div class="prediction-box">Predicted Rating Count (Sales Volume): '
                f'{pred_count:,}</div>',
                unsafe_allow_html=True,
            )

            # Interpretation
            if pred_count > 50000:
                tier = "🔥 **Blockbuster** — expected viral/bestseller volume"
            elif pred_count > 10000:
                tier = "✅ **High Seller** — strong market demand expected"
            elif pred_count > 2000:
                tier = "📊 **Moderate Seller** — average market traction"
            else:
                tier = "⚠️ **Low Seller** — niche product or low demand"

            st.info(f"**Sales Tier**: {tier}")

            # Gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pred_count,
                    title={"text": "Predicted Rating Count"},
                    gauge={
                        "axis": {"range": [0, 200000]},
                        "bar": {"color": "#FF9900"},
                        "steps": [
                            {"range": [0, 2000], "color": "#ffebcc"},
                            {"range": [2000, 10000], "color": "#ffc266"},
                            {"range": [10000, 50000], "color": "#ff9900"},
                            {"range": [50000, 200000], "color": "#cc7a00"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": pred_count,
                        },
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Similar products
            st.subheader("🔍 Similar Products in Dataset")
            similar = df[df["main_category"] == sel_cat][
                ["product_name", "discounted_price", "actual_price",
                 "discount_percentage", "rating", "rating_count"]
            ].sort_values("rating_count", ascending=False).head(5)
            st.dataframe(similar.reset_index(drop=True), use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 5 — DATA TABLE
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("📋 Cleaned Dataset")

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            cat_filter = st.multiselect(
                "Filter by Category",
                options=sorted(df["main_category"].unique()),
                default=[],
            )
        with fc2:
            min_r, max_r = st.slider(
                "Filter by Rating Range",
                min_value=float(df["rating"].min()),
                max_value=float(df["rating"].max()),
                value=(float(df["rating"].min()), float(df["rating"].max())),
                step=0.1,
            )

        view_df = df.copy()
        if cat_filter:
            view_df = view_df[view_df["main_category"].isin(cat_filter)]
        view_df = view_df[
            (view_df["rating"] >= min_r) & (view_df["rating"] <= max_r)
        ]

        st.write(f"Showing **{len(view_df):,}** products")
        st.dataframe(
            view_df[[
                "product_name", "main_category",
                "discounted_price", "actual_price",
                "discount_percentage", "rating", "rating_count",
            ]].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        csv_data = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data as CSV",
            data=csv_data,
            file_name="amazon_filtered.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
