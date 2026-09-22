"""
frontend.py  –  Streamlit frontend for House Rent Prediction
Run with:  streamlit run frontend.py
Talks to the Flask backend at http://localhost:5000
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BACKEND = "http://localhost:5000"

# ---------- page config ----------
st.set_page_config(
    page_title="🏠 House Rent Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-title { font-size: 1rem; opacity: 0.9; }
    .prediction-value { font-size: 3rem; font-weight: 800; margin: 0.5rem 0; }
    .prediction-range { font-size: 0.9rem; opacity: 0.85; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stSidebar"] { background: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


# ---------- load dropdown options ----------
@st.cache_data(ttl=3600)
def get_options():
    try:
        r = requests.get(f"{BACKEND}/api/options", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback defaults
    return {
        "area_type":        ["Super Area", "Carpet Area", "Built Area"],
        "city":             ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"],
        "furnishing_status":["Unfurnished", "Semi-Furnished", "Furnished"],
        "tenant_preferred": ["Bachelors/Family", "Bachelors", "Family"],
        "floor_type":       ["Ground", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                             "Upper Basement", "Lower Basement"],
    }

OPTIONS = get_options()

# ---------- header ----------
st.markdown('<p class="main-header">🏠 House Rent Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered rent estimation for Indian cities</p>', unsafe_allow_html=True)

# ---------- sidebar inputs ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/house.png", width=80)
    st.title("Property Details")
    st.markdown("---")

    bhk       = st.selectbox("🛏 BHK (Bedrooms)",       options=[1, 2, 3, 4, 5, 6], index=1)
    bathroom  = st.selectbox("🚿 Bathrooms",             options=[1, 2, 3, 4, 5, 6], index=1)
    size      = st.slider("📐 Size (sq ft)",             min_value=100, max_value=8000,
                          value=1000, step=50)
    city      = st.selectbox("🏙 City",                  options=OPTIONS["city"])
    area_type = st.selectbox("🗺 Area Type",             options=OPTIONS["area_type"])
    furnishing= st.selectbox("🛋 Furnishing Status",     options=OPTIONS["furnishing_status"])
    tenant    = st.selectbox("👥 Tenant Preferred",      options=OPTIONS["tenant_preferred"])

    st.markdown("**Floor Details**")
    col1, col2 = st.columns(2)
    with col1:
        floor_num = st.selectbox("Floor",        options=OPTIONS["floor_type"], index=0)
    with col2:
        total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=3)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Rent", use_container_width=True, type="primary")

# ---------- main content ----------
col_main, col_insight = st.columns([1, 1], gap="large")

with col_main:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        payload = {
            "bhk":               bhk,
            "size":              size,
            "area_type":         area_type,
            "city":              city,
            "furnishing_status": furnishing,
            "tenant_preferred":  tenant,
            "bathroom":          bathroom,
            "floor_num":         floor_num,
            "total_floors":      total_floors,
        }
        with st.spinner("Calculating …"):
            try:
                r = requests.post(f"{BACKEND}/api/predict", json=payload, timeout=10)
                if r.status_code == 200:
                    result = r.json()
                    rent  = result["predicted_rent"]
                    low   = result["range_low"]
                    high  = result["range_high"]

                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-title">Estimated Monthly Rent</div>
                        <div class="prediction-value">₹{rent:,.0f}</div>
                        <div class="prediction-range">Range: ₹{low:,.0f} – ₹{high:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rent,
                        title={"text": "Predicted Rent (₹)"},
                        gauge={
                            "axis": {"range": [0, max(high * 1.5, 100000)]},
                            "bar":  {"color": "#667eea"},
                            "steps": [
                                {"range": [0, low],   "color": "#dcfce7"},
                                {"range": [low, high],"color": "#fef9c3"},
                                {"range": [high, max(high * 1.5, 100000)], "color": "#fee2e2"},
                            ],
                            "threshold": {
                                "line": {"color": "#764ba2", "width": 4},
                                "thickness": 0.75,
                                "value": rent,
                            },
                        },
                        number={"prefix": "₹", "valueformat": ",.0f"},
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Summary metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Low Estimate",  f"₹{low:,.0f}")
                    c2.metric("Prediction",    f"₹{rent:,.0f}")
                    c3.metric("High Estimate", f"₹{high:,.0f}")

                    # store for insights tab
                    st.session_state["last_result"] = result
                    st.session_state["last_payload"] = payload

                else:
                    st.error(f"Backend error: {r.json().get('error', r.text)}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to backend. Make sure `python app.py` is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    else:
        st.info("👈  Fill in the property details in the sidebar and click **Predict Rent**.")

with col_insight:
    st.subheader("📈 Market Insights")

    # City-wise average rent chart (hard-coded sample from dataset)
    city_avg = {
        "Mumbai":    45000,
        "Delhi":     22000,
        "Bangalore": 20000,
        "Hyderabad": 16000,
        "Chennai":   14000,
        "Kolkata":   12000,
    }
    df_city = pd.DataFrame(list(city_avg.items()), columns=["City", "Avg Rent (₹)"])
    fig_city = px.bar(
        df_city, x="City", y="Avg Rent (₹)",
        color="Avg Rent (₹)", color_continuous_scale="Viridis",
        title="Average Rent by City",
        text_auto=True,
    )
    fig_city.update_layout(
        height=300, margin=dict(t=50, b=30, l=20, r=20),
        coloraxis_showscale=False,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_city, use_container_width=True)

    # Furnishing impact
    furn_mult = {
        "Unfurnished": 1.0,
        "Semi-Furnished": 1.25,
        "Furnished": 1.6,
    }
    if "last_result" in st.session_state:
        base = st.session_state["last_result"]["predicted_rent"]
        base_furn = st.session_state["last_payload"]["furnishing_status"]
        base_mult = furn_mult[base_furn]
        rows = [
            {"Furnishing": k, "Est. Rent (₹)": round(base / base_mult * v, -2)}
            for k, v in furn_mult.items()
        ]
        df_furn = pd.DataFrame(rows)
        fig_furn = px.pie(
            df_furn, names="Furnishing", values="Est. Rent (₹)",
            title="Furnishing vs Rent (this property)",
            color_discrete_sequence=["#a5b4fc", "#818cf8", "#4f46e5"],
        )
        fig_furn.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_furn, use_container_width=True)
    else:
        # BHK impact chart
        bhk_avg = {1: 10000, 2: 18000, 3: 30000, 4: 48000, 5: 65000}
        df_bhk = pd.DataFrame(list(bhk_avg.items()), columns=["BHK", "Avg Rent (₹)"])
        fig_bhk = px.line(
            df_bhk, x="BHK", y="Avg Rent (₹)",
            markers=True, title="BHK vs Average Rent",
            color_discrete_sequence=["#667eea"],
        )
        fig_bhk.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20),
                               plot_bgcolor="white")
        st.plotly_chart(fig_bhk, use_container_width=True)

# ---------- feature importance (below) ----------
st.markdown("---")
st.subheader("🔬 Model Feature Importance")
feat_imp = {
    "Total_Floors":      0.231,
    "City":              0.210,
    "Bathroom":          0.201,
    "Size":              0.194,
    "BHK":               0.097,
    "Floor_Num":         0.022,
    "Furnishing Status": 0.021,
    "Area Type":         0.013,
    "Tenant Preferred":  0.010,
}
df_fi = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
df_fi.sort_values("Importance", ascending=True, inplace=True)
fig_fi = px.bar(
    df_fi, x="Importance", y="Feature", orientation="h",
    color="Importance", color_continuous_scale="Blues",
    title="What influences rent the most?",
    text_auto=".0%",
)
fig_fi.update_layout(
    height=350, margin=dict(t=50, b=20, l=20, r=20),
    coloraxis_showscale=False, plot_bgcolor="white",
)
st.plotly_chart(fig_fi, use_container_width=True)

# ---------- footer ----------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.8rem;'>"
    "🏠 House Rent Prediction · Powered by RandomForest + Flask + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
