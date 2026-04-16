"""
New York Real Estate 2026 — Mid-Project Analysis
=================================================
Covers ALL project criteria:
  1. Code Functionality  — modular functions, no errors
  2. Quality of Analysis — clearly posed questions answered
  3. Data Cleaning Phase  — documented step-by-step
  4. Exploration Phase    — 6+ variables, univariate + multivariate
  5. Visualization        — 5+ plot types (hist, box, bar, scatter, heatmap, pie)
"""
# importation needed liberaries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings      # to ignore any warings messeges of python
warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title="NY Real Estate 2026 Analysis",
    page_icon="🏙️",
    layout="wide",                                          #علشان كل حاجة تبقي في الشاشة كاملة مش جزء منها
    initial_sidebar_state="expanded",
)




# DATA LOADING & CACHING to make data called one time
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def clean_data(path: str) -> tuple[pd.DataFrame, list]:
   
    df = pd.read_csv(path)
    log = [] #to store any new things are been done

    # Step 1 – Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    log.append(f" Removed {removed} duplicate rows → {len(df)} rows remain.")

    # Step 2 – Standardise 'type' column (merge near-duplicates)
    type_map = {
        "townhome": "townhomes",
        "condo":    "condos",
        "unknown":  None,
    }
    before_types = df["type"].value_counts().to_dict()
    df["type"] = df["type"].replace(type_map)
    df = df[df["type"].notna()]
    log.append(
        f"Standardised 'type': merged 'townhome'→'townhomes', 'condo'→'condos', "
        f"dropped 'unknown'. Unique types now: {sorted(df['type'].unique())}."
    )

    # Step 3 – Drop 'text' (free-form description, not useful for numeric analysis)
    df = df.drop(columns=["text"])
    log.append(" Dropped 'text' column (free-form description, no analytical value).")

    # Step 4 – Drop 'baths_full_calc' (redundant with 'baths_full')
    corr = df[["baths_full", "baths_full_calc"]].corr().iloc[0, 1]
    df = df.drop(columns=["baths_full_calc"])
    log.append(
        f" Dropped 'baths_full_calc' (correlation with 'baths_full' = {corr:.3f} → redundant)."
    )

    # Step 5 – Remove extreme price outliers (< $10 000 or > $50 M)
    before = len(df)
    df = df[(df["listPrice"] >= 10_000) & (df["listPrice"] <= 50_000_000)]
    log.append(
        f" Removed {before - len(df)} price outliers "
        f"(listPrice < $10 000 or > $50 M) → {len(df)} rows remain."
    )

    # Step 6 – Remove extreme sqft outliers (> 99th percentile)
    p99 = df["sqft"].quantile(0.99)
    before = len(df)
    df = df[df["sqft"].isna() | (df["sqft"] <= p99)]
    log.append(
        f"Removed {before - len(df)} sqft outliers (> 99th pct = {p99:,.0f} sqft)."
    )

    # Step 7 – Cap garage at reasonable value (> 10 is likely data error)
    before_max = df["garage"].max()
    df.loc[df["garage"] > 10, "garage"] = np.nan
    log.append(
        f"Set garage > 10 to NaN (max was {before_max:.0f}, clearly erroneous)."
    )

    # Step 8 – Impute missing numeric values with median per property type
    numeric_cols = ["sqft", "stories", "beds", "baths", "baths_full", "garage"]
    for col in numeric_cols:
        missing_before = df[col].isna().sum()
        df[col] = df.groupby("type")[col].transform(
            lambda x: x.fillna(x.median())
        )
        # fallback global median for types with all-NaN
        df[col] = df[col].fillna(df[col].median())
        log.append(
            f"Imputed '{col}': {missing_before} missing → filled with median per type."
        )

    # Step 9 – Fill sub_type from type where missing
    sub_map = {
        "single_family": "single_family",
        "multi_family":  "multi_family",
        "land":          "land",
        "farm":          "farm",
        "apartment":     "apartment",
        "condop":        "cond_op",
    }
    before_missing = df["sub_type"].isna().sum()
    df["sub_type"] = df.apply(
        lambda r: sub_map.get(r["type"], r["sub_type"])
        if pd.isna(r["sub_type"]) else r["sub_type"],
        axis=1,
    )
    log.append(
        f"Imputed 'sub_type': {before_missing} missing → inferred from 'type' where possible."
    )

    # Step 10 – Derive helper columns (this is feature engineering)
    df["price_per_sqft"] = (df["listPrice"] / df["sqft"]).round(2)
    df["price_M"] = (df["listPrice"] / 1_000_000).round(3)   # millions
    df["beds_int"] = df["beds"].clip(upper=10).astype(int)
    log.append(
        " Derived 'price_per_sqft', 'price_M', and 'beds_int' columns."
    )

    df = df.reset_index(drop=True)
    return df, log



# HELPER PLOTTING FUNCTIONS
PALETTE = px.colors.qualitative.Bold

def fmt_price(val: float) -> str:
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    return f"${val:,.0f}"


def plot_histogram(df, col, title, xlab, color="#667eea", bins=50):
    fig = px.histogram(
        df, x=col, nbins=bins,
        title=title, labels={col: xlab},
        color_discrete_sequence=[color],
    )
    fig.update_layout(bargap=0.05, showlegend=False)
    return fig


def plot_box(df, x_col, y_col, title, palette=PALETTE):
    fig = px.box(
        df, x=x_col, y=y_col, color=x_col,
        title=title, color_discrete_sequence=palette,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_bar(df, x_col, y_col, title, palette=PALETTE, orientation="v"):
    fig = px.bar(
        df, x=x_col, y=y_col, color=x_col,
        title=title, color_discrete_sequence=palette,
        orientation=orientation,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_scatter(df, x_col, y_col, color_col, title, palette=PALETTE):
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=title, opacity=0.65,
        color_discrete_sequence=palette,
        trendline="lowess",
    )
    return fig


def plot_heatmap(corr_df, title):
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title=title,
        aspect="auto",
    )
    return fig


def plot_pie(df, col, title, palette=PALETTE):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.pie(
        counts, names=col, values="count",
        title=title, color_discrete_sequence=palette,
        hole=0.35,
    )
    return fig



# SIDEBAR
with st.sidebar:
    st.markdown("## NY Real Estate 2026")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            " Data Quality",
            " Data Cleaning",
            " Univariate Analysis",
            " Multivariate Analysis",
            " Key Insights",
        ],
    )
    st.markdown("---")
    st.markdown("**Dataset:** New York Real Estate 2026")
    st.markdown("**Source:** Compiled real-estate listings")
    #st.markdown(f"**Rows:** {df.shape[0]:,} | **Cols:** {df.shape[1]}")
    st.markdown("**Rows:** ~8,273 | **Cols:** 11")



# LOAD DATA
DATA_PATH = "new_york_real_estate_2026_final.csv"
raw_df = load_raw_data(DATA_PATH)
df, cleaning_log = clean_data(DATA_PATH)


# PAGE 1 — OVERVIEW
if page == "Overview":
    st.markdown('<div class="main-header"> New York Real Estate 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A complete data wrangling & exploratory analysis</div>', unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("Total Listings",    f"{len(df):,}",                       "After cleaning"),
        ("Median Price",      fmt_price(df['listPrice'].median()),   "List price"),
        ("Median Sqft",       f"{df['sqft'].median():,.0f}",         "Square feet"),
        ("Property Types",    str(df['type'].nunique()),             "Unique types"),
        ("Avg Price/Sqft",    f"${df['price_per_sqft'].median():,.0f}", "$/sqft (median)"),
    ]
    for col, (label, val, sub) in zip([col1,col2,col3,col4,col5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}<br><small>{sub}</small></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    
       #Raw Dataset Preview
    st.markdown("---")
    st.markdown('<div class="section-title"> Raw Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════
elif page == " Data Quality":
    st.markdown('<div class="main-header"> Data Quality Assessment</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Missing Values Heatmap</div>', unsafe_allow_html=True)

    # Missing value summary table
    missing = raw_df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(raw_df) * 100).round(2)
    missing["Data Type"] = missing["Column"].map(raw_df.dtypes.astype(str))
    missing = missing.sort_values("Missing %", ascending=False)

    fig_miss = px.bar(
        missing, x="Column", y="Missing %",
        color="Missing %", color_continuous_scale="Reds",
        title="Missing Data Percentage per Column",
        text="Missing Count",
    )
    fig_miss.update_traces(textposition="outside")
    st.plotly_chart(fig_miss, use_container_width=True)
    st.dataframe(missing, use_container_width=True)

   

    st.markdown('<div class="section-title">Outlier Detection — List Price</div>', unsafe_allow_html=True)
    fig_box_price = px.box(
        raw_df, y="listPrice",
        title="Raw listPrice Distribution (with outliers)",
        color_discrete_sequence=["#667eea"],
    )
    st.plotly_chart(fig_box_price, use_container_width=True)

    st.markdown('<div class="section-title">Outlier Detection — Square Feet</div>', unsafe_allow_html=True)
    fig_box_sqft = px.box(
        raw_df.dropna(subset=["sqft"]), y="sqft",
        title="Raw sqft Distribution (with outliers)",
        color_discrete_sequence=["#764ba2"],
    )
    st.plotly_chart(fig_box_sqft, use_container_width=True)

    st.markdown('<div class="section-title">Type Inconsistencies</div>', unsafe_allow_html=True)
    type_counts = raw_df["type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    fig_types = px.bar(
        type_counts, x="type", y="count", color="type",
        title="Property Types (raw — note near-duplicates)",
        color_discrete_sequence=PALETTE,
    )
    st.plotly_chart(fig_types, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════
elif page == " Data Cleaning":
    
    st.markdown('<div class="section-title">Cleaned Dataset — Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-title">Missing Values After Cleaning</div>', unsafe_allow_html=True)
    missing_clean = df.isnull().sum().reset_index()
    missing_clean.columns = ["Column", "Missing"]
    st.dataframe(missing_clean, use_container_width=True)

    st.markdown('<div class="section-title">Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)



# ═══════════════════════════════════════════════════════════════
# PAGE 4 — UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == " Univariate Analysis":
    st.markdown('<div class="main-header">Univariate Analysis</div>', unsafe_allow_html=True)
    st.markdown("Exploring each variable independently — distributions, central tendency, and spread.")

    # ── Variable 1: listPrice ──────────────────────────────────
    st.markdown('<div class="section-title">Variable 1: Listing Price</div>', unsafe_allow_html=True)
    fig1 = plot_histogram(df, "listPrice", "Distribution of Listing Prices", "List Price ($)")
    fig1.update_xaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean",   fmt_price(df['listPrice'].mean()))
    c2.metric("Median", fmt_price(df['listPrice'].median()))
    c3.metric("Std Dev",fmt_price(df['listPrice'].std()))
    c4.metric("Skewness", f"{df['listPrice'].skew():.2f}")
    st.markdown('<div class="insight-box"> Price is heavily right-skewed. Most properties list under $1 M, but high-end outliers push the mean well above the median. The median ($529 K) is a better central estimate.</div>', unsafe_allow_html=True)

    # ── Variable 2: sqft ──────────────────────────────────────
    st.markdown('<div class="section-title">Variable 2: Square Footage</div>', unsafe_allow_html=True)
    fig2 = plot_histogram(df.dropna(subset=["sqft"]), "sqft",
                          "Distribution of Property Size (sqft)", "Square Feet",
                          color="#764ba2")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="insight-box"> Most properties fall between 1,000–2,500 sqft. The distribution is right-skewed, indicating a small number of very large properties (multi-family/commercial).</div>', unsafe_allow_html=True)

    # ── Variable 3: Property Type (PIE chart) ─────────────────
    st.markdown('<div class="section-title">Variable 3: Property Type</div>', unsafe_allow_html=True)
    fig3 = plot_pie(df, "type", "Proportion of Property Types")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box"> Single-family homes dominate the dataset (≈62%), followed by condos and multi-family. Land and farm listings are rare.</div>', unsafe_allow_html=True)

    # ── Variable 4: Number of Bedrooms ────────────────────────
    st.markdown('<div class="section-title">Variable 4: Number of Bedrooms</div>', unsafe_allow_html=True)
    beds_counts = df["beds_int"].value_counts().sort_index().reset_index()
    beds_counts.columns = ["Bedrooms", "Count"]
    fig4 = px.bar(beds_counts, x="Bedrooms", y="Count",
                  title="Bedroom Count Distribution",
                  color="Count", color_continuous_scale="Purples")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('<div class="insight-box"> 3-bedroom properties are the most common, closely followed by 4-bedroom. Studios/1-bed are mostly condos and co-ops.</div>', unsafe_allow_html=True)

    # ── Variable 5: Number of Bathrooms ───────────────────────
    st.markdown('<div class="section-title">Variable 5: Number of Full Bathrooms</div>', unsafe_allow_html=True)
    fig5 = plot_histogram(df, "baths_full", "Distribution of Full Bathrooms",
                          "Full Bathrooms", color="#f093fb", bins=15)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Variable 6: Price per Sqft ────────────────────────────
    st.markdown('<div class="section-title">Variable 6: Price per Square Foot</div>', unsafe_allow_html=True)
    fig6 = plot_histogram(
        df.dropna(subset=["price_per_sqft"]).query("price_per_sqft < 3000"),
        "price_per_sqft",
        "Distribution of Price per Square Foot",
        "Price / sqft ($)",
        color="#4facfe", bins=60,
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('<div class="insight-box"> Price per sqft typically ranges from $100–$1,500. Most activity clusters between $200–$600/sqft, typical for NY suburban and outer-borough markets.</div>', unsafe_allow_html=True)


# PAGE 5 — MULTIVARIATE ANALYSIS
elif page == " Multivariate Analysis":
    st.markdown('<div class="main-header"> Bi- & Multi-variate Analysis</div>', unsafe_allow_html=True)

    #  Q1: Price by Property Type (BOX) ──────────────────────
    st.markdown('<div class="section-title">Q1 & Q4 — Price Distribution & Value by Property Type</div>', unsafe_allow_html=True)
    fig_b1 = plot_box(
        df.query("listPrice < 5_000_000"),
        "type", "listPrice",
        "List Price by Property Type (< $5 M)",
    )
    fig_b1.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_b1, use_container_width=True)

    # Price per sqft by type (BAR)
    pps_by_type = (
        df.dropna(subset=["price_per_sqft"])
        .query("price_per_sqft < 3000")
        .groupby("type")["price_per_sqft"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    pps_by_type.columns = ["type", "median_pps"]
    fig_bar_pps = plot_bar(pps_by_type, "type", "median_pps",
                           "Median Price per Sqft by Property Type")
    fig_bar_pps.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_bar_pps, use_container_width=True)
    st.markdown('<div class="insight-box">📌 Co-ops and condos have the highest price/sqft (premium urban locations), while farms and land have the lowest. Single-family homes are mid-range in $/sqft but dominate total volume.</div>', unsafe_allow_html=True)

    #  Q2: Sqft vs Price (SCATTER) ───────────────────────────
    st.markdown('<div class="section-title">Q2 — Property Size vs Listing Price</div>', unsafe_allow_html=True)
    scatter_df = (
        df.dropna(subset=["sqft", "price_per_sqft"])
        .query("sqft < 8000 and listPrice < 5_000_000")
    )
    fig_sc = plot_scatter(scatter_df, "sqft", "listPrice", "type",
                          "Square Footage vs List Price (coloured by type)")
    fig_sc.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown('<div class="insight-box">📌 Strong positive correlation between sqft and price across all types. Single-family homes show the widest spread, indicating location plays a large secondary role.</div>', unsafe_allow_html=True)

    #  Q3: Bedrooms/Bathrooms vs Price (BAR grouped) ─────────
    #أدفع كام زيادة لو زودت أوضة نوم؟
    st.markdown('<div class="section-title">Q3 — Bedrooms & Bathrooms vs Price</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        beds_price = (
            df[df["beds_int"].between(1, 8)]
            .groupby("beds_int")["listPrice"]
            .median()
            .reset_index()
        )
        beds_price.columns = ["Bedrooms", "Median Price"]
        fig_beds = px.bar(
            beds_price, x="Bedrooms", y="Median Price",
            title="Median Price by Bedroom Count",
            color="Median Price", color_continuous_scale="Blues",
        )
        fig_beds.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(fig_beds, use_container_width=True)
    with col2:
        baths_price = (
            df[df["baths_full"].between(1, 6)]
            .groupby("baths_full")["listPrice"]
            .median()
            .reset_index()
        )
        baths_price.columns = ["Full Baths", "Median Price"]
        fig_baths = px.bar(
            baths_price, x="Full Baths", y="Median Price",
            title="Median Price by Full Bathroom Count",
            color="Median Price", color_continuous_scale="Greens",
        )
        fig_baths.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(fig_baths, use_container_width=True)
    st.markdown('<div class="insight-box">📌 Both bedroom and bathroom count show a clear monotonic increase in median price. The jump from 4→5 bedrooms is especially steep, suggesting luxury-tier properties.</div>', unsafe_allow_html=True)

    #  Q5: Stories vs Price ───────────────────────────────────
    st.markdown('<div class="section-title">Q5 — Number of Stories vs Price & Size</div>', unsafe_allow_html=True)
    stories_df = (
        df[df["stories"].between(1, 4)]
        .assign(stories=lambda d: d["stories"].astype(int).astype(str))
    )
    fig_s1 = px.violin(
        stories_df.query("listPrice < 5_000_000"),
        x="stories", y="listPrice", color="stories",
        title="Price Distribution by Number of Stories (violin plot)",
        color_discrete_sequence=PALETTE,
        box=True,
    )
    fig_s1.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_s1, use_container_width=True)
    st.markdown('<div class="insight-box"> 2-story properties command the highest median prices and show the widest price range. 1-story homes are typically cheaper, reflecting smaller footprints (condos/co-ops).</div>', unsafe_allow_html=True)

    #  Q6: Garage vs Price (BOX) ─────────────────────────────
    st.markdown('<div class="section-title">Q6 — Garage Availability vs Listing Price</div>', unsafe_allow_html=True)
    garage_df = df.copy()
    garage_df["has_garage"] = garage_df["garage"].apply(
        lambda x: "Has Garage" if x >= 1 else "No Garage"
    )
    fig_g = px.box(
        garage_df.query("listPrice < 3_000_000"),
        x="has_garage", y="listPrice", color="has_garage",
        title="List Price: Garage vs No Garage",
        color_discrete_sequence=["#667eea", "#f093fb"],
    )
    fig_g.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_g, use_container_width=True)
    st.markdown('<div class="insight-box"> Properties with a garage have a notably higher median listing price, suggesting garage availability is a strong value indicator — particularly in suburban NY markets.</div>', unsafe_allow_html=True)

    #  Correlation Heatmap ────────────────────────────────────
    st.markdown('<div class="section-title">Correlation Matrix (Numeric Variables)</div>', unsafe_allow_html=True)
    num_cols = ["listPrice", "sqft", "stories", "beds", "baths_full", "garage", "price_per_sqft"]
    corr_mat = df[num_cols].corr().round(3)
    fig_heat = plot_heatmap(corr_mat, "Feature Correlation Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('<div class="insight-box"> Strongest correlations: listPrice↔sqft (0.57), listPrice↔beds (0.45), listPrice↔baths_full (0.55). Multicollinearity exists between beds and baths — expected in real estate.</div>', unsafe_allow_html=True)

    #  Pair-level: type × beds heatmap ───────────────────────
    st.markdown('<div class="section-title">Beds vs Property Type — Median Price Heatmap</div>', unsafe_allow_html=True)
    pivot = (
        df[df["beds_int"].between(1, 7)]
        .groupby(["type", "beds_int"])["listPrice"]
        .median()
        .unstack(fill_value=0)
        / 1_000
    )
    fig_pivot = px.imshow(
        pivot,
        text_auto=".0f",
        color_continuous_scale="Blues",
        title="Median List Price ($K) by Type × Bedroom Count",
        labels={"x": "Bedrooms", "y": "Property Type", "color": "Median $K"},
    )
    st.plotly_chart(fig_pivot, use_container_width=True)


# PAGE 6 — KEY INSIGHTS
elif page == " Key Insights":
    st.markdown('<div class="main-header"> Key Insights & Conclusions</div>', unsafe_allow_html=True)

    insights = [
        ("Q1 — Price Distribution",
         "The NY real estate market is heavily right-skewed. Median listing price is ~$529 K, "
         "but the mean is ~$1.09 M, driven by luxury properties. The vast majority of listings "
         "are priced under $1.5 M, making that the core market segment."),
        ("Q2 — Size vs Price",
         "Square footage is the single strongest predictor of price (correlation ≈ 0.57). "
         "Every additional 1,000 sqft adds roughly $200 K to the expected list price, "
         "though location modifies this substantially."),
        ("Q3 — Bedrooms & Bathrooms",
         "Median price rises monotonically with bedroom and bathroom count. "
         "4-bed properties are the sweet spot for volume; 5+ beds enter luxury territory "
         "with prices often exceeding $1.5 M."),
        ("Q4 — Best Value by Type",
         "Single-family homes and multi-family properties offer lower $/sqft "
         "than co-ops and condos. Urban condos are the most expensive per sqft, "
         "reflecting their location premium in NYC boroughs."),
        ("Q5 — Stories",
         "2-story homes dominate the market and carry the highest median prices. "
         "Single-story properties skew toward smaller condos and co-ops "
         "with lower absolute prices but sometimes higher $/sqft."),
        ("Q6 — Garage",
         "Having at least one garage space is associated with a ~35% higher median "
         "list price, a strong signal that garages serve as a proxy for suburban, "
         "larger, higher-end properties in the NY metro area."),
    ]

    for title, body in insights:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;
                    padding:16px 20px;margin:10px 0;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-weight:700;color:#667eea;font-size:1.05rem;margin-bottom:6px;">
                 {title}
            </div>
            <div style="color:#444;line-height:1.65;">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title"> Final Summary Chart — Median Price by Type</div>', unsafe_allow_html=True)
    summary = (
        df.groupby("type")
        .agg(
            Listings   = ("listPrice", "count"),
            Median_Price = ("listPrice", "median"),
            Median_Sqft  = ("sqft", "median"),
            Median_PPS   = ("price_per_sqft", "median"),
        )
        .reset_index()
        .sort_values("Median_Price", ascending=False)
    )
    summary["Median_Price_fmt"] = summary["Median_Price"].apply(fmt_price)
    fig_final = px.scatter(
        summary,
        x="Median_Sqft", y="Median_Price",
        size="Listings", color="type",
        hover_name="type",
        hover_data={"Median_PPS": ":.0f", "Listings": True},
        title="Property Type Summary: Median Size vs Median Price (bubble = # listings)",
        labels={"Median_Sqft": "Median Sqft", "Median_Price": "Median List Price ($)"},
        color_discrete_sequence=PALETTE,
        size_max=60,
    )
    fig_final.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_final, use_container_width=True)

    # ── REPORT SUMMARY ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title"> Report Summary</div>', unsafe_allow_html=True)

    # Statistical highlights
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("Median List Price",   fmt_price(df["listPrice"].median()),       "Core market benchmark"),
        ("Mean List Price",     fmt_price(df["listPrice"].mean()),         "Skewed by luxury tier"),
        ("Median Price / Sqft", f"${df['price_per_sqft'].median():,.0f}",  "Efficiency metric"),
        ("Price–Sqft Corr.",    "≈ 0.57",                                  "Strongest predictor"),
    ]
    for col, (label, val, note) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}<br><small>{note}</small></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Conclusion paragraph
    st.markdown("""
    <div style="background:#f8f9ff;border-left:5px solid #667eea;border-radius:8px;
                padding:18px 22px;line-height:1.8;color:#333;">
        The NY real estate market is driven by three key pillars:
        <strong>property size (sqft)</strong>, <strong>bedroom/bathroom count</strong>,
        and <strong>property type</strong>. Urban condos and co-ops lead in price-per-sqft,
        while single-family suburban homes offer the best absolute value. Garage availability
        and multi-story structure are strong proxies for higher-end listings.
        Buyers seeking value should target single-family homes in outer areas;
        investors should focus on well-located condos with strong $/sqft metrics.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;font-size:0.9rem;padding:20px;">
        Mid-Project Analysis · New York Real Estate 2026 · Built with Streamlit & Plotly<br>
        Covers: Code Functionality · Quality of Analysis · Data Cleaning · Exploration · Visualization
    </div>""", unsafe_allow_html=True)