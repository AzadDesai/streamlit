import streamlit as st
import pandas as pd
import numpy as np
import math

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Nifty/BankNifty Option Analyzer", layout="wide")
st.title("📊 True Data-Driven Option Analyzer")

# --- 2. MATH BRAIN (Replaces Scipy) ---
# This fixes the "ModuleNotFoundError" by using standard Python math
def norm_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# --- 3. SMART FILE LOADER ---
def load_data(file):
    try:
        # Strategy A: Try reading with header on Row 2 (Standard NSE)
        df = pd.read_csv(file, header=1)
        
        # Verify if we found "STRIKE" column
        if not any('STRIKE' in str(c).upper() for c in df.columns):
            # Strategy B: Try reading normally (header Row 1)
            file.seek(0)
            df = pd.read_csv(file, header=0)
            
        # Strategy C: If still failing, brute-force find the header
        if not any('STRIKE' in str(c).upper() for c in df.columns):
            file.seek(0)
            df_raw = pd.read_csv(file, header=None)
            header_idx = -1
            for i, row in df_raw.head(10).iterrows():
                row_text = row.astype(str).str.upper().str.cat(sep=' ')
                if 'STRIKE' in row_text and 'LTP' in row_text:
                    header_idx = i
                    break
            if header_idx != -1:
                df = df_raw.iloc[header_idx+1:].copy()
                df.columns = df_raw.iloc[header_idx]
            else:
                st.error("❌ Could not find 'STRIKE' column. Please check CSV.")
                return None

        # CLEANUP: Fix column names and numbers
        df.columns = df.columns.astype(str).str.strip().str.upper()
        
        # Remove empty rows
        df = df.dropna(subset=['STRIKE'])
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# --- 4. ANALYSIS ENGINE ---
def analyze(df):
    try:
        # 1. Identify Columns (Handles duplicates like LTP, LTP.1)
        cols = df.columns.tolist()
        
        # Find 'STRIKE'
        col_strike = next((c for c in cols if 'STRIKE' in c), None)
        if not col_strike:
            st.error("Column 'STRIKE' not found!")
            return None, None
            
        # Separate Calls (Left) and Puts (Right)
        # In NSE CSV: Calls are first, Puts are last
        ltp_cols = [c for c in cols if 'LTP' in c]
        iv_cols = [c for c in cols if 'IV' in c]
        
        if len(ltp_cols) < 2 or len(iv_cols) < 2:
            st.error("Could not find Call/Put LTP columns.")
            return None, None
            
        c_ltp, p_ltp = ltp_cols[0], ltp_cols[-1]
        c_iv, p_iv = iv_cols[0], iv_cols[-1]
        
        # 2. Extract & Clean Data
        data = pd.DataFrame()
        
        def clean(x):
            if isinstance(x, str):
                x = x.replace(',', '').replace('-', '0').replace('"', '')
                try: return float(x)
                except: return 0.0
            return x
            
        data['strike'] = df[col_strike].apply(clean)
        data['c_ltp'] = df[c_ltp].apply(clean)
        data['p_ltp'] = df[p_ltp].apply(clean)
        data['c_iv'] = df[c_iv].apply(clean)
        data['p_iv'] = df[p_iv].apply(clean)
        
        # Filter Junk
        data = data[(data['c_ltp'] > 0) & (data['p_ltp'] > 0) & (data['strike'] > 0)]
        
        # 3. Calculate Spot (Call = Put)
        data['diff'] = abs(data['c_ltp'] - data['p_ltp'])
        atm_row = data.loc[data['diff'].idxmin()]
        spot = atm_row['strike'] + (atm_row['c_ltp'] - atm_row['p_ltp'])
        
        return spot, data
        
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return None, None

def calculate_greeks_and_trade(row, spot, type_):
    T = 1.0/365.0 # 1 Day
    r = 0.10
    K = row['strike']
    
    if type_ == 'Call':
        price = row['c_ltp']
        iv = row['c_iv']/100 or 0.2
        d1 = (np.log(spot/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        delta = norm_cdf(d1)
        theta = (- (spot * iv * norm_pdf(d1)) / (2 * np.sqrt(T))) / 365.0
    else:
        price = row['p_ltp']
        iv = row['p_iv']/100 or 0.2
        d1 = (np.log(spot/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        delta = norm_cdf(d1) - 1
        theta = (- (spot * iv * norm_pdf(d1)) / (2 * np.sqrt(T))) / 365.0
        
    # Trade Logic
    daily_move = (spot * iv) / 19.1
    risk = daily_move * abs(delta)
    sl = price - risk
    tp = price + (risk * 2.0)
    
    decay_hours = (price * 0.10 / abs(theta)) * 6.0 if abs(theta) > 0 else 0
    
    return {
        "Strike": f"{K}",
        "Entry": round(price, 2),
        "SL": round(sl, 2),
        "TP": round(tp, 2),
        "Max Hold": f"{round(decay_hours, 1)} Hours",
        "Reason": f"IV: {round(iv*100,1)}% | Delta: {round(delta,2)}"
    }

# --- 5. UI MAIN ---
uploaded = st.file_uploader("📂 Upload Option Chain CSV", type="csv")

if uploaded:
    df = load_data(uploaded)
    if df is not None:
        spot, data = analyze(df)
        if spot:
            st.success(f"✅ Spot Price Detected: **{round(spot, 2)}**")
            
            # Find ATM
            data['dist'] = abs(data['strike'] - spot)
            atm = data.sort_values('dist').iloc[0]
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🚀 Call Setup")
                res = calculate_greeks_and_trade(atm, spot, 'Call')
                st.metric("Strike", res['Strike'])
                st.metric("Entry", res['Entry'])
                st.metric("Stop Loss", res['SL'])
                st.metric("Target", res['TP'])
                st.info(f"Hold Limit: {res['Max Hold']}")
                st.caption(res['Reason'])
                
            with c2:
                st.subheader("📉 Put Setup")
                res = calculate_greeks_and_trade(atm, spot, 'Put')
                st.metric("Strike", res['Strike'])
                st.metric("Entry", res['Entry'])
                st.metric("Stop Loss", res['SL'])
                st.metric("Target", res['TP'])
                st.info(f"Hold Limit: {res['Max Hold']}")
                st.caption(res['Reason'])
