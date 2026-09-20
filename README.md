# 📦 Amazon Sales Prediction

> **Predict product sales volume on Amazon using Machine Learning**  
> Built with Python · Streamlit · scikit-learn · Plotly

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd amazon-sales-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure amazon.csv is in the same folder as app.py

# 5. Launch the app
streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**

---

## 📁 Project Structure

```
amazon-sales-prediction/
│
├── amazon.csv           # Raw Amazon product dataset
├── app.py               # Main application (ML backend + Streamlit UI)
├── requirements.txt     # Python dependencies
├── project_report.md    # Detailed project report
└── README.md            # This file
```

---

## 🧠 What It Does

| Feature | Description |
|---------|------------|
| **Data Ingestion** | Loads and cleans `amazon.csv` (₹ prices, % discounts, rating counts) |
| **Feature Engineering** | Extracts top-level category, label encodes, log-transforms target |
| **Multi-Model Training** | Random Forest, Gradient Boosting, Ridge, Linear Regression |
| **Model Evaluation** | MAE, RMSE, R², 5-Fold Cross-Validated R² |
| **Interactive Prediction** | Input product details → instant sales volume forecast |
| **Visualisations** | Histograms, scatter plots, heatmap, feature importance, gauge chart |
| **Data Table** | Filterable, downloadable view of the cleaned dataset |

---

## 🖥️ App Tabs

| Tab | Content |
|-----|---------|
| 🏠 **Overview** | Project summary, pipeline steps, target variable explanation |
| 📈 **EDA & Insights** | 6 interactive Plotly charts exploring the dataset |
| 🤖 **Model Performance** | Metrics table, R² comparison, Actual vs Predicted scatter |
| 🔮 **Predict Sales** | Form to input new product → predicted rating count + gauge |
| 📋 **Data Table** | Filtered, downloadable product table |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `amazon.csv` |
| Source | https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset |
| Records | ~1,465 products |
| Target variable | `rating_count` (proxy for sales volume) |

### Features Used for Prediction

| Feature | Description |
|---------|-------------|
| `discounted_price` | Current selling price (₹) |
| `actual_price` | Original MRP (₹) |
| `discount_percentage` | Percentage discount offered |
| `rating` | Average star rating (1–5) |
| `main_category` | Top-level product category (label-encoded) |

---

## 🤖 Models

| Model | Library | Notes |
|-------|---------|-------|
| **Random Forest** | scikit-learn | Best overall performance (~R² 0.55) |
| **Gradient Boosting** | scikit-learn | Close second, robust to outliers |
| **Ridge Regression** | scikit-learn | Fast, regularised linear baseline |
| **Linear Regression** | scikit-learn | Simple baseline for comparison |

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
plotly==5.22.0
joblib==1.4.2
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔮 Sales Volume Tiers

| Tier | Predicted Count | Meaning |
|------|----------------|---------|
| 🔥 Blockbuster | > 50,000 | Viral / bestseller |
| ✅ High Seller | 10,001 – 50,000 | Strong demand |
| 📊 Moderate Seller | 2,001 – 10,000 | Average traction |
| ⚠️ Low Seller | ≤ 2,000 | Niche / low demand |

---

## ⚠️ Notes

- `rating_count` is used as a **proxy for sales volume** — not actual transaction data.
- All models are trained and cached on first launch; subsequent runs are fast.
- The app uses **log1p** transformation on the target for better model fit.
- Predictions are back-transformed with **expm1** before display.

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).  
Dataset is used for educational purposes only.

---

## 🙏 Acknowledgements

- Dataset: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
- Framework: [Streamlit](https://streamlit.io/)
- ML: [scikit-learn](https://scikit-learn.org/)
- Charts: [Plotly](https://plotly.com/python/)
