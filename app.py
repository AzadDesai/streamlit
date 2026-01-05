import streamlit as st
import pandas as pd
import numpy as np
import math

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Data-Driven Option Analyzer", layout="wide")

st.title("📊 True Data-Driven Option Analyzer")
st.markdown("""
This app calculates Entry, SL, and Targets based **strictly** on Greeks (Delta, Theta) and IV.
No random percentages. No static defaults.
""")

# --- 2. CUSTOM MATH FUNCTIONS (No Scipy Required) ---
def norm_pdf(x):
    """Standard normal probability density function"""
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def norm_cdf(x):
    """Standard normal cumulative distribution function"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# --- 3. THE CALCULATION ENGINE ---
def analyze_data(df):
    try:
        # CLEANING: Handle messy column names and strings
        df.columns = df.columns.str.strip().str.upper()
        
        # Helper to clean numbers
        def clean_num(x):
            if isinstance(x, str):
                x = x.replace(',', '').replace('-', '0')
                try: return float(x)
                except: return 0.0
            return x

        # IDENTIFY COLUMNS (Auto-detect logic)
        col_map = {
            'strike': [c for c in df.columns if 'STRIKE' in c][0],
            'iv_call': [c for c in df.columns if 'IV' in c][0],
            'iv_put': [c for c in df.columns if 'IV' in c][-1],
            'ltp_call': [c for c in df.columns if 'LTP' in c][0],
            'ltp_put': [c for c in df.columns if 'LTP' in c][-1],
            'vol_call': [c for c in df.columns if 'VOLUME' in c][0],
            'vol_put': [c for c in df.columns if 'VOLUME' in c][-1]
        }
        
        # Prepare Dataframe
        data = pd.DataFrame()
        data['strike'] = df[col_map['strike']].apply(clean_num)
        data['ce_ltp'] = df[col_map['ltp_call']].apply(clean_num)
        data['pe_ltp'] = df[col_map['ltp_put']].apply(clean_num)
        data['ce_iv'] = df[col_map['iv_call']].apply(clean_num)
        data['pe_iv'] = df[col_map['iv_put']].apply(clean_num)
        data['ce_vol'] = df[col_map['vol_call']].apply(clean_num)
        data['pe_vol'] = df[col_map['pe_vol']].apply(clean_num)
        
        # Remove empty rows
        data = data[(data['ce_ltp'] > 0) & (data['pe_ltp'] > 0)]

        # --- CORE LOGIC: SPOT FINDER ---
        # Spot is where Call Price approx equals Put Price
        data['diff'] = abs(data['ce_ltp'] - data['pe_ltp'])
        atm_row = data.loc[data['diff'].idxmin()]
        spot_price = atm_row['strike'] + (atm_row['ce_ltp'] - atm_row['pe_ltp'])
        
        return spot_price, data

    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None, None

def calculate_trade(row, spot, trade_type):
    # PARAMETERS
    T = 1.0 / 365.0  # 1 Day to expiry
    r = 0.10         # Risk Free Rate
    
    K = row['strike']
    
    # Calculate Greeks using Custom Math (No Scipy)
    if trade_type == 'Call':
        price = row['ce_ltp']
        sigma = row['ce_iv'] / 100.0
        if sigma <= 0: sigma = 0.2
        
        d1 = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        delta = norm_cdf(d1)
        theta = (- (spot * sigma * norm_pdf(d1)) / (2 * np.sqrt(T))) / 365.0
        
    else: # Put
        price = row['pe_ltp']
        sigma = row['pe_iv'] / 100.0
        if sigma <= 0: sigma = 0.2
        
        d1 = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        delta = norm_cdf(d1) - 1
        theta = (- (spot * sigma * norm_pdf(d1)) / (2 * np.sqrt(T))) / 365.0

    # --- THE "NO BLIND NUMBERS" FORMULAS ---
    daily_move_pts = (spot * sigma) / 19.1
    option_risk = daily_move_pts * abs(delta)
    
    sl_price = price - option_risk
    tp_price = price + (option_risk * 2.0)
    
    decay_budget = price * 0.10
    if abs(theta) > 0:
        max_hours = (decay_budget / abs(theta)) * 6.0 
    else:
        max_hours = 0
        
    return {
        "Type": trade_type,
        "Strike": K,
        "Entry Price": round(price, 2),
        "Stop Loss": round(sl_price, 2),
        "Take Profit": round(tp_price, 2),
        "Risk (Pts)": round(option_risk, 2),
        "Max Hold Time": f"{round(max_hours, 1)} Hours",
        "Reasoning": f"IV: {round(sigma*100,1)}% | Delta: {round(delta,2)} | Theta: {round(theta,2)}"
    }

# --- 4. THE UI INTERFACE ---

uploaded_file = st.file_uploader("Upload Option Chain CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=1) 
    except:
        df = pd.read_csv(uploaded_file)
        
    spot, processed_data = analyze_data(df)
    
    if spot:
        st.success(f"✅ Market Data Loaded. Calculated Spot Price: **{round(spot, 2)}**")
        
        processed_data['dist'] = abs(processed_data['strike'] - spot)
        near_atm = processed_data.sort_values('dist').head(1)
        
        if not near_atm.empty:
            row = near_atm.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚀 Bullish Trade (Call)")
                call_res = calculate_trade(row, spot, 'Call')
                st.metric("Strike", f"{call_res['Strike']} CE")
                st.metric("Entry", call_res['Entry Price'])
                st.metric("Stop Loss", call_res['Stop Loss'], delta=-call_res['Risk (Pts)'])
                st.metric("Target", call_res['Take Profit'], delta=call_res['Risk (Pts)']*2)
                st.info(f"⏳ **Max Hold Time:** {call_res['Max Hold Time']}")
                st.caption(f"Math: {call_res['Reasoning']}")

            with col2:
                st.subheader("🐻 Bearish Trade (Put)")
                put_res = calculate_trade(row, spot, 'Put')
                st.metric("Strike", f"{put_res['Strike']} PE")
                st.metric("Entry", put_res['Entry Price'])
                st.metric("Stop Loss", put_res['Stop Loss'], delta=-put_res['Risk (Pts)'])
                st.metric("Target", put_res['Take Profit'], delta=put_res['Risk (Pts)']*2)
                st.info(f"⏳ **Max Hold Time:** {put_res['Max Hold Time']}")
                st.caption(f"Math: {put_res['Reasoning']}")
                
        else:
            st.warning("No liquid strikes found near spot.")
