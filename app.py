import streamlit as st
import pandas as pd
import numpy as np
import math

# --- 1. SETUP ---
st.set_page_config(page_title="Nifty Option Analyzer", layout="wide")
st.title("📊 True Data-Driven Option Analyzer")

# --- 2. MATH FUNCTIONS (Replaces Scipy) ---
def norm_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# --- 3. ROBUST DATA LOADER ---
def load_and_clean_csv(uploaded_file):
    try:
        # ATTEMPT 1: Try reading with header on Row 2 (Standard for NSE files)
        df = pd.read_csv(uploaded_file, header=1)
        
        # Check if we found the columns
        cols = df.columns.str.upper()
        if not any('STRIKE' in c for c in cols):
            # ATTEMPT 2: Try standard read if Attempt 1 failed
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        
        # Normalize Column Names
        df.columns = df.columns.str.strip().str.upper()
        
        # Helper to clean currency text (e.g., "1,200.50" or "-")
        def clean_num(x):
            if isinstance(x, str):
                x = x.replace(',', '').replace('"', '').replace("'", "")
                if x.strip() == '-' or x.strip() == '':
                    return 0.0
                try: return float(x)
                except: return 0.0
            return x

        # Map Columns specifically for your file structure
        # Your file has: ... LTP, ..., STRIKE, ..., LTP.1, ...
        all_cols = df.columns.tolist()
        
        col_strike = [c for c in all_cols if 'STRIKE' in c][0]
        
        # Call Side (Left side of file)
        col_ltp_call = [c for c in all_cols if 'LTP' in c][0] # First LTP
        col_iv_call  = [c for c in all_cols if 'IV' in c][0]  # First IV
        
        # Put Side (Right side of file, usually has .1 suffix)
        col_ltp_put = [c for c in all_cols if 'LTP' in c][-1] # Last LTP
        col_iv_put  = [c for c in all_cols if 'IV' in c][-1]  # Last IV
        
        # Create Clean DataFrame
        data = pd.DataFrame()
        data['strike'] = df[col_strike].apply(clean_num)
        data['ce_ltp'] = df[col_ltp_call].apply(clean_num)
        data['pe_ltp'] = df[col_ltp_put].apply(clean_num)
        data['ce_iv'] = df[col_iv_call].apply(clean_num)
        data['pe_iv'] = df[col_iv_put].apply(clean_num)
        
        # Filter: Remove rows where Price is 0
        data = data[(data['ce_ltp'] > 0) & (data['pe_ltp'] > 0)]
        
        return data

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

# --- 4. ANALYSIS ENGINE ---
def analyze_market(data):
    # 1. Calculate Spot Price (Call Parity)
    # Find row where Call Price is closest to Put Price
    data['diff'] = abs(data['ce_ltp'] - data['pe_ltp'])
    atm_row = data.loc[data['diff'].idxmin()]
    spot_price = atm_row['strike'] + (atm_row['ce_ltp'] - atm_row['pe_ltp'])
    
    return spot_price, atm_row

def calculate_trade_setup(row, spot, trade_type):
    # Constants
    T = 1.0 / 365.0 # 1 Day to Expiry
    r = 0.10        # Risk Free Rate
    K = row['strike']
    
    if trade_type == 'Call':
        price = row['ce_ltp']
        sigma = row['ce_iv'] / 100.0 if row['ce_iv'] > 0 else 0.2
        d1 = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm_cdf(d1)
        theta = (- (spot * sigma * norm_pdf(d1)) / (2 * np.sqrt(T))) / 365.0
    else:
        price = row['pe_ltp']
        sigma = row['pe_iv'] / 100.0 if row['pe_iv'] > 0 else 0.2
        d1 = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm_cdf(d1) - 1
        theta = (- (spot * sigma * norm_pdf(d1)) / (2 * np.sqrt(T))) / 365.0

    # Dynamic Rules
    daily_move = (spot * sigma) / 19.1
    risk = daily_move * abs(delta)
    
    sl = price - risk
    tp = price + (risk * 2.0)
    
    decay_budget = price * 0.10 # Max 10% decay allowed
    max_hours = (decay_budget / abs(theta)) * 6.0 if abs(theta) > 0 else 0
        
    return {
        "Strike": f"{K}",
        "Type": trade_type,
        "Entry": round(price, 2),
        "SL": round(sl, 2),
        "TP": round(tp, 2),
        "Max Hold": f"{round(max_hours, 1)} Hours",
        "Reason": f"IV: {round(sigma*100,1)}% | Delta: {round(delta,2)}"
    }

# --- 5. UI DISPLAY ---
uploaded_file = st.file_uploader("📂 Upload Option Chain CSV", type=['csv'])

if uploaded_file is not None:
    data = load_and_clean_csv(uploaded_file)
    
    if data is not None and not data.empty:
        spot, atm_row = analyze_market(data)
        st.success(f"✅ Market Identified! Calculated Spot Price: **{round(spot, 2)}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Call (Bullish) Setup")
            res = calculate_trade_setup(atm_row, spot, 'Call')
            st.metric("Strike", f"{res['Strike']} CE")
            st.metric("Entry Price", res['Entry'])
            st.metric("Stop Loss", res['SL'], delta=-round(res['Entry']-res['SL'], 1))
            st.metric("Target", res['TP'], delta=round(res['TP']-res['Entry'], 1))
            st.warning(f"⏳ **Max Hold Time:** {res['Max Hold']}")
            st.caption(f"Stats: {res['Reason']}")

        with col2:
            st.subheader("📉 Put (Bearish) Setup")
            res = calculate_trade_setup(atm_row, spot, 'Put')
            st.metric("Strike", f"{res['Strike']} PE")
            st.metric("Entry Price", res['Entry'])
            st.metric("Stop Loss", res['SL'], delta=-round(res['Entry']-res['SL'], 1))
            st.metric("Target", res['TP'], delta=round(res['TP']-res['Entry'], 1))
            st.warning(f"⏳ **Max Hold Time:** {res['Max Hold']}")
            st.caption(f"Stats: {res['Reason']}")
            
    else:
        st.error("Could not extract valid data. Please check the CSV format.")
