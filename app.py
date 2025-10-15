import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pytesseract
import pdfplumber
import json
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Chromatogram Analyzer", layout="wide")

# --- Users ---
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

st.sidebar.success(f"Logged in as: {st.session_state.user}")

# --- Uploads & Inputs ---
uploaded_file = st.sidebar.file_uploader("Upload CSV, PDF, PNG or JPG", type=["csv","pdf","png","jpg","jpeg"])
start_time = st.sidebar.number_input("Start Time", value=0.0, step=0.1)
end_time = st.sidebar.number_input("End Time", value=10.0, step=0.1)
cal_file = st.sidebar.file_uploader("Calibration CSV (optional)", type=["csv"])

# --- Functions ---
def read_csv_smart(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 2:
            st.error(f"CSV must have exactly 2 columns, got {df.shape[1]}")
            return None
        df.columns = ["Time","Signal"]
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None
    return df

def extract_image_data(input_data):
    import re
    try:
        if isinstance(input_data, Image.Image):
            img = input_data
        else:
            img = Image.open(input_data).convert("RGB")
        text = pytesseract.image_to_string(img)
        rows = []
        for line in text.splitlines():
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line.replace(',', '.'))
            if len(nums) >= 2:
                try:
                    rows.append([float(nums[0]), float(nums[1])])
                except:
                    continue
        df = pd.DataFrame(rows, columns=["Time","Signal"])
        return df if not df.empty else None
    except:
        return None

def extract_pdf_ocr(uploaded_pdf):
    all_rows = []
    uploaded_pdf.seek(0)
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=150).original
            df_page = extract_image_data(img)
            if df_page is not None:
                all_rows.extend(df_page.values.tolist())
    if all_rows:
        df = pd.DataFrame(all_rows, columns=["Time","Signal"])
        df = df.drop_duplicates().sort_values("Time").reset_index(drop=True)
        return df
    return None

# --- Main ---
df = None
image_displayed = False

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = read_csv_smart(uploaded_file)
    elif uploaded_file.name.lower().endswith((".png","jpg","jpeg")):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_column_width=True)
        image_displayed = True
        df = extract_image_data(img)
    elif uploaded_file.name.lower().endswith(".pdf"):
        st.info("Displaying PDF first page as image")
        with pdfplumber.open(uploaded_file) as pdf:
            page = pdf.pages[0]
            img = page.to_image(resolution=150).original
            st.image(img, caption="PDF page as image", use_column_width=True)
            image_displayed = True
            df = extract_pdf_ocr(uploaded_file)

    if df is None or df.empty:
        st.warning("No numeric data detected in this file.")
        if st.button("Convert Graph to CSV"):
            st.info("Graphical extraction not yet implemented. Use an external tool to digitize the plot.")
        st.stop()

    # --- Select zone ---
    mask = (df["Time"]>=start_time) & (df["Time"]<=end_time)
    df_zone = df.loc[mask]
    if df_zone.empty:
        st.error("No data in the selected zone. Adjust start/end times.")
        st.stop()

    # --- Calculations ---
    noise_std = df_zone["Signal"].std()
    sn_usp = df_zone["Signal"].max()/noise_std
    sn_norm = df_zone["Signal"].mean()/noise_std
    lod = 3*noise_std
    loq = 10*noise_std

    lod_conc, loq_conc = lod, loq
    if cal_file:
        try:
            cal_df = pd.read_csv(cal_file)
            X = cal_df.iloc[:,0].values.reshape(-1,1)
            Y = cal_df.iloc[:,1].values
            slope = LinearRegression().fit(X,Y).coef_[0]
            lod_conc = lod/slope
            loq_conc = loq/slope
        except:
            st.warning("Calibration file could not be read or used.")

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Signal"], mode="lines", name="Signal"))
    fig.add_trace(go.Scatter(x=df_zone["Time"], y=df_zone["Signal"], mode="lines", name="Selected Zone", line=dict(color="red")))
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics ---
    metrics = pd.DataFrame({
        "Metric":["S/N USP","S/N Normal","LOD (signal)","LOQ (signal)","LOD (conc.)","LOQ (conc.)"],
        "Value":[sn_usp,sn_norm,lod,loq,lod_conc,loq_conc]
    })
    st.table(metrics)