# 🏙️ New York Real Estate 2026 — Mid-Project Analysis

An interactive data wrangling and exploratory data analysis (EDA) app built with **Streamlit** and **Plotly**.

## 📌 Project Criteria Covered

| Criterion | Details |
|---|---|
| ✅ Code Functionality | Modular functions, fully reproducible, zero errors |
| ✅ Quality of Analysis | 6 clearly stated research questions, fully answered |
| ✅ Data Cleaning Phase | 10 documented cleaning steps with before/after metrics |
| ✅ Exploration Phase | 6+ variables — univariate + bi/multivariate analysis |
| ✅ Visualization | Histogram, Box, Bar, Scatter, Pie, Heatmap, Violin (7 plot types) |

## 📊 Dataset

**New York Real Estate 2026** — 8,273 real estate listings with 11 columns:
`type, sub_type, text, listPrice, sqft, stories, beds, baths, baths_full, baths_full_calc, garage`

**Quality issues present in raw data:**
- Missing values in 7/11 columns (up to 77% missing in some)
- Duplicate rows
- Inconsistent type labels (`condo` vs `condos`, `townhome` vs `townhomes`)
- Price outliers ($1 listings, $80 M listings)
- Extreme sqft values (1.48 M sqft)
- Nonsensical garage values (1440 spaces)

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔬 Research Questions

1. How are listing prices distributed across New York?
2. What is the relationship between property size and price?
3. How do bedrooms/bathrooms affect price?
4. Which property types offer best value (price/sqft)?
5. How do multi-story properties compare to single-story?
6. Is garage availability associated with higher prices?

## 📁 Files

- `app.py` — Main Streamlit application
- `new_york_real_estate_2026_final.csv` — Dataset
- `requirements.txt` — Dependencies
- `.streamlit/config.toml` — App theme
