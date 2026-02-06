import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# --- PAGE CONFIG ---
st.set_page_config(page_title="Thursday Sniper 🦅", page_icon="🎯", layout="centered")

# Custom CSS for Dark Mode & Metrics
st.markdown("""
<style>
    .big-metric { font-size: 2.5em; font-weight: bold; color: #4ade80; }
    .risk-metric { font-size: 1.2em; color: #facc15; }
    .stApp { background-color: #0e1117; color: #fafafa; }
    .card { background-color: #262730; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    try:
        model.load_model("thursday_model.json")
        return model, True
    except:
        return None, False

model, loaded = load_model()

st.title("🦅 EURUSD Thursday Sniper")
st.caption("Quantum Volatility Predictor | AI-Powered by XGBoost")

if not loaded:
    st.error("❌ Model not found! Please upload 'thursday_model.json' to the repository.")
    st.stop()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📝 Weekly Data Input")
    st.info("Enter data from Mon-Wed Close")
    
    with st.expander("Monday Data", expanded=True):
        r_mon = st.number_input("Mon Range (Pips)", value=60.0, step=1.0)
        b_mon_raw = st.number_input("Mon Body (Close-Open)", value=10.0, step=1.0, help="Positive=Green, Negative=Red")
    
    with st.expander("Tuesday Data", expanded=True):
        r_tue = st.number_input("Tue Range (Pips)", value=70.0, step=1.0)
        b_tue_raw = st.number_input("Tue Body (Close-Open)", value=-20.0, step=1.0)
    
    with st.expander("Wednesday Data", expanded=True):
        r_wed = st.number_input("Wed Range (Pips)", value=55.0, step=1.0)
        b_wed = st.number_input("Wed Body (Absolute Pips)", value=30.0, step=1.0)
    
    st.markdown("---")
    # Auto ADR Estimate
    est_adr = (r_mon + r_tue + r_wed) / 3
    adr_val = st.number_input("ADR (5 Days)", value=est_adr, step=1.0, help="If unknown, keep the auto-calculated value.")
    
    week_num = st.selectbox("Week of Month", [1, 2, 3, 4], index=1)
    
    predict_btn = st.button("🦅 SNIPE TARGET", type="primary", use_container_width=True)

# --- MAIN LOGIC ---
if predict_btn:
    # 1. Calculate Inputs
    b_mon = abs(b_mon_raw)
    b_tue = abs(b_tue_raw)
    
    features = [
        'Range_Monday', 'Range_Tuesday', 'Range_Wednesday',
        'Body_Monday', 'Body_Tuesday', 'Body_Wednesday',
        'ADR_5_Wednesday', 'Week_Num_Wednesday'
    ]
    
    input_df = pd.DataFrame([[r_mon, r_tue, r_wed, b_mon, b_tue, b_wed, adr_val, week_num]], columns=features)
    
    # 2. Predict (Log -> Exp)
    pred_log = model.predict(input_df)[0]
    pred_pips = np.expm1(pred_log)
    
    # 3. Determine Bias
    bias_score = 0
    if b_mon_raw > 0: bias_score += 1
    if b_tue_raw > 0: bias_score += 1
    if b_mon_raw < 0: bias_score -= 1
    if b_tue_raw < 0: bias_score -= 1
    
    if bias_score >= 1: 
        bias_text = "🟢 BULLISH (Buy Dips)"
        bias_color = "#4ade80"
    elif bias_score <= -1: 
        bias_text = "🔴 BEARISH (Sell Rallies)"
        bias_color = "#f87171"
    else: 
        bias_text = "🟡 NEUTRAL / CHOPPY"
        bias_color = "#facc15"

    # 4. Display Results
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🧭 Direction Bias")
        st.markdown(f"<h3 style='color:{bias_color};'>{bias_text}</h3>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Market Context")
        st.metric("ADR Strength", f"{adr_val:.0f} Pips")

    st.markdown("### 🎯 Volatility Targets")
    
    c1, c2, c3 = st.columns(3)
    
    median_error = 18.0 # Based on our deep analysis
    safe_zone = pred_pips - median_error
    max_zone = pred_pips + median_error
    
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("🛡️ SAFE BANKING", f"{safe_zone:.0f} Pips", delta="High Probability")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="card" style="border: 1px solid #4ade80;">', unsafe_allow_html=True)
        st.metric("🦅 AI TARGET", f"{pred_pips:.0f} Pips", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("🚀 MAX EXTENSION", f"{max_zone:.0f} Pips", delta="Low Probability", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

    if pred_pips > 100:
        st.warning("🔥 HIGH VOLATILITY ALERT: Wide Stops Required!")
    elif pred_pips < 60:
        st.info("💤 Low Volatility Expected. Scalping conditions.")

else:
    st.markdown("### 👋 Marhba ya Sniper!")
    st.markdown("""
    Use the sidebar on the left to enter Wednesday's closing data.
    The AI will calculate the optimal **Thursday Expansion Range**.
    
    **How to use:**
    1. Wait for Wednesday Candle Close.
    2. Input Mon/Tue/Wed Range & Body.
    3. Click **SNIPE**.
    """)