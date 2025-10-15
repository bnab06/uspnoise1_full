import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pytesseract
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Chromatogram Analyzer", layout="wide")

# --- Users ---
import json, os
USERS_FILE = "users.json"
with open(USERS_FILE, "r") as f:
    users = json.load(f)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

# --- Sidebar Login ---
st.sidebar.title("Login")
if not st.session_state.logged_in:
    selected_user = st.sidebar.selectbox("User", list(users.keys()))
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if selected_user in users and password == users[selected_user]['pwd']:
            st.session_state.logged_in = True
            st.session_state.user = selected_user
        else:
            st.sidebar.error("Incorrect credentials")
    st.title("Chromatogram Analyzer – USP S/N Method B")
    st.info("Please log in from the sidebar.")
    st.stop()

# --- Main UI ---
st.sidebar.success(f"Logged in as: {st.session_state.user}")
uploaded_file = st.sidebar.file_uploader("Upload CSV, PDF, or Image", type=["csv","pdf","png","jpg","jpeg"])
start_time = st.sidebar.number_input("Start Time", value=0.0, step=0.1)
end_time = st.sidebar.number_input("End Time", value=10.0, step=0.1)
cal_file = st.sidebar.file_uploader("Calibration CSV (optional)", type=["csv"])

def read_csv_smart(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = ["Time","Signal"]
    return df

def extract_image_data(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    text = pytesseract.image_to_string(img)
    import re
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line.replace(',', '.'))
        if len(nums)>=2:
            try:
                rows.append([float(nums[0]),float(nums[1])])
            except: continue
    df = pd.DataFrame(rows, columns=["Time","Signal"])
    return df

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = read_csv_smart(uploaded_file)
    else:
        df = extract_image_data(uploaded_file)

    mask = (df["Time"]>=start_time) & (df["Time"]<=end_time)
    df_zone = df.loc[mask]
    noise_std = df_zone["Signal"].std()
    sn_usp = df_zone["Signal"].max()/noise_std
    sn_norm = df_zone["Signal"].mean()/noise_std
    lod = 3*noise_std
    loq = 10*noise_std

    # Calibration
    lod_conc, loq_conc = lod, loq
    if cal_file:
        cal_df = pd.read_csv(cal_file)
        X = cal_df.iloc[:,0].values.reshape(-1,1)
        Y = cal_df.iloc[:,1].values
        slope = LinearRegression().fit(X,Y).coef_[0]
        lod_conc = lod/slope
        loq_conc = loq/slope

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Signal"], mode="lines", name="Signal"))
    fig.add_trace(go.Scatter(x=df_zone["Time"], y=df_zone["Signal"], mode="lines", name="Selected Zone"))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    metrics = pd.DataFrame({
        "Metric":["S/N USP","S/N Normal","LOD (signal)","LOQ (signal)","LOD (conc.)","LOQ (conc.)"],
        "Value":[sn_usp,sn_norm,lod,loq,lod_conc,loq_conc]
    })
    st.table(metrics)
