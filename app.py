import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pdfplumber
import cv2
import io
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

def digitize_image(img):
    """Convert an image of chromatogram into DataFrame"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_cv, 200, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return None
    # Group by X (horizontal) and take max Y
    df = pd.DataFrame(coords, columns=["Y","X"])
    df = df.groupby("X")["Y"].min().reset_index()
    df["Time"] = np.interp(df["X"], [df["X"].min(), df["X"].max()], [0,10])
    df["Signal"] = np.interp(df["Y"], [df["Y"].min(), df["Y"].max()], [0, df["Y"].max()])
    df = df[["Time","Signal"]]
    return df

def extract_pdf_image(uploaded_pdf):
    uploaded_pdf.seek(0)
    with pdfplumber.open(uploaded_pdf) as pdf:
        page = pdf.pages[0]
        img = page.to_image(resolution=200).original
        return img

# --- Main ---
df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = read_csv_smart(uploaded_file)
    elif uploaded_file.name.lower().endswith((".png","jpg","jpeg")):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_column_width=True)
        df = digitize_image(img)
    elif uploaded_file.name.lower().endswith(".pdf"):
        img = extract_pdf_image(uploaded_file)
        st.image(img, caption="PDF first page", use_column_width=True)
        df = digitize_image(img)

    if df is None or df.empty:
        st.warning("No numeric data could be extracted. Use a CSV or high-contrast image.")
        st.stop()

    # --- Select zone ---
    mask = (df["Time"]>=start_time) & (df["Time"]<=end_time)
    df_zone = df.loc[mask]
    if df_zone.empty:
        st.error("No data in the selected zone. Adjust start/end times.")
        st.stop()

    # --- Calculations ---
    noise_std = df_zone["Signal"].std()
    sn_usp = df_zone["Signal"].max()/noise_std if noise_std>0 else 0
    sn_norm = df_zone["Signal"].mean()/noise_std if noise_std>0 else 0
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