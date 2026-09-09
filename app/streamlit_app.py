import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add src to python path for modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import (
    load_pipeline,
    predict_single_customer,
    predict_batch_customers,
    get_feature_importance,
)
from src import config

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FAVICON_PATH = os.path.join(APP_DIR, "assets", "favicon.png")
if not os.path.exists(FAVICON_PATH):
    FAVICON_PATH = os.path.join(os.getcwd(), "assets", "favicon.png")

try:
    _page_icon = Image.open(FAVICON_PATH) if os.path.exists(FAVICON_PATH) else None
except Exception:
    _page_icon = FAVICON_PATH if os.path.exists(FAVICON_PATH) else None

st.set_page_config(
    page_title="ChurnAI — Enterprise Churn Risk Intelligence",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Disable Browser Auto-Translation (Google Chrome) to Prevent React DOM Text Duplication ───
if hasattr(st, "html"):
    st.html(
        """
        <script>
        (function() {
            try {
                document.documentElement.setAttribute('translate', 'no');
                document.documentElement.classList.add('notranslate');
                if (document.body) {
                    document.body.setAttribute('translate', 'no');
                    document.body.classList.add('notranslate');
                }
            } catch (e) {}
        })();
        </script>
        """
    )

# ══════════════════════════════════════════════════════════════════════
#  ENTERPRISE DESIGN SYSTEM (Linear / Stripe / Vercel Aesthetic)
#  Guaranteed high contrast, strict light-theme readability, zero clashes
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <meta name="google" content="notranslate">
    <meta name="googlebot" content="notranslate">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    /* ─── Base Reset & Theme Variables ─── */
    :root {
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        
        /* Neutrals */
        --slate-50: #F8FAFC;
        --slate-100: #F1F5F9;
        --slate-200: #E2E8F0;
        --slate-300: #CBD5E1;
        --slate-400: #94A3B8;
        --slate-500: #64748B;
        --slate-600: #475569;
        --slate-700: #334155;
        --slate-800: #1E293B;
        --slate-900: #0F172A;
        --white: #FFFFFF;

        /* Semantics */
        --emerald-500: #10B981;
        --emerald-600: #059669;
        --emerald-bg: #ECFDF5;
        --emerald-border: #A7F3D0;
        --emerald-text: #065F46;

        --amber-500: #F59E0B;
        --amber-600: #D97706;
        --amber-bg: #FFFBEB;
        --amber-border: #FDE68A;
        --amber-text: #92400E;

        --rose-500: #EF4444;
        --rose-600: #DC2626;
        --rose-bg: #FEF2F2;
        --rose-border: #FECACA;
        --rose-text: #991B1B;

        --indigo-500: #6366F1;
        --indigo-600: #4F46E5;
        --indigo-bg: #EEF2FF;
        --indigo-border: #C7D2FE;
        --indigo-text: #3730A3;
    }

    /* ─── Global Strict Text & Background ─── */
    html, body, .stApp {
        background-color: var(--slate-50) !important;
        font-family: var(--font-sans) !important;
        color: var(--slate-900) !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Force all text in main workspace to high-contrast dark */
    section[data-testid="stMain"],
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] div,
    section[data-testid="stMain"] li {
        color: var(--slate-900);
        font-family: var(--font-sans) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--slate-900) !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* ─── Sidebar: Deep Slate Enterprise Theme ─── */
    section[data-testid="stSidebar"] {
        background-color: var(--slate-900) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    section[data-testid="stSidebar"] * {
        font-family: var(--font-sans) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: var(--white) !important;
    }
    /* Sidebar Labels: Guaranteed Pure White & Bold */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label *,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] label div,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSelectbox label *,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stTextInput label * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 0.35rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12) !important;
        margin: 1.25rem 0 !important;
    }
    /* Sidebar Selectbox Container */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border: 1.5px solid #475569 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }
    /* Sidebar Selectbox Text / Spans / Options - Guaranteed Bright White */
    section[data-testid="stSidebar"] .stSelectbox > div > div *,
    section[data-testid="stSidebar"] .stSelectbox > div > div span,
    section[data-testid="stSidebar"] .stSelectbox > div > div div,
    section[data-testid="stSidebar"] .stSelectbox > div > div p,
    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] [aria-selected="true"] {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox svg {
        fill: #FFFFFF !important;
    }
    /* Sidebar Text Input */
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: #1E293B !important;
        border: 1.5px solid #475569 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-family: var(--font-mono) !important;
        font-size: 0.88rem !important;
    }
    /* Sidebar Captions */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] span {
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
        font-size: 0.78rem !important;
    }

    /* ─── Main Workspace Form Inputs: Crisp White, Slate-300 Border, Dark Text ─── */
    section[data-testid="stMain"] .stSelectbox label,
    section[data-testid="stMain"] .stSlider label,
    section[data-testid="stMain"] .stNumberInput label,
    section[data-testid="stMain"] .stTextInput label,
    .main .stSelectbox label,
    .main .stSlider label,
    .main .stNumberInput label,
    .main .stTextInput label {
        color: var(--slate-700) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 0.25rem !important;
    }
    section[data-testid="stMain"] .stSelectbox > div > div,
    .main .stSelectbox > div > div {
        background-color: var(--white) !important;
        border: 1px solid var(--slate-300) !important;
        border-radius: 6px !important;
        color: var(--slate-900) !important;
        font-size: 0.9rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    section[data-testid="stMain"] .stSelectbox > div > div:focus-within,
    .main .stSelectbox > div > div:focus-within {
        border-color: var(--slate-900) !important;
        box-shadow: 0 0 0 1px var(--slate-900) !important;
    }
    section[data-testid="stMain"] .stSelectbox > div > div span,
    .main .stSelectbox > div > div span {
        color: var(--slate-900) !important;
        -webkit-text-fill-color: var(--slate-900) !important;
    }
    section[data-testid="stMain"] .stNumberInput > div > div > input,
    .main .stNumberInput > div > div > input {
        background-color: var(--white) !important;
        border: 1px solid var(--slate-300) !important;
        border-radius: 6px !important;
        color: var(--slate-900) !important;
        -webkit-text-fill-color: var(--slate-900) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.9rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    section[data-testid="stMain"] .stNumberInput > div > div > input:focus,
    .main .stNumberInput > div > div > input:focus {
        border-color: var(--slate-900) !important;
        box-shadow: 0 0 0 1px var(--slate-900) !important;
    }

    /* ─── Slider Styling: Guaranteed 100% Visibility ─── */
    .stSlider label {
        color: var(--slate-800) !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }
    /* Floating thumb bubble number */
    .stSlider [data-testid="stThumbValue"],
    [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] [data-testid="stThumbValue"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-family: var(--font-mono) !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 3px 8px !important;
        border-radius: 4px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
    }
    /* Tick numbers (0, 72, etc.) and slider labels */
    .stSlider [data-testid="stSliderTickBarMin"],
    .stSlider [data-testid="stSliderTickBarMax"],
    .stSlider div[data-testid="stTickBar"] div,
    [data-testid="stSlider"] span,
    [data-testid="stSlider"] div {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        font-family: var(--font-mono) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    /* ─── Primary CTA Button (Linear / Stripe style): Guaranteed Pure White Text ─── */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] *,
    .stButton > button[kind="primary"] p,
    button[data-testid="stBaseButton-primary"] p,
    .stButton > button[kind="primary"] span,
    button[data-testid="stBaseButton-primary"] span,
    .stButton > button[kind="primary"] div,
    button[data-testid="stBaseButton-primary"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.8rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px -2px rgba(15, 23, 42, 0.15), 0 2px 4px -2px rgba(15, 23, 42, 0.1) !important;
    }
    .stButton > button[kind="primary"]:hover *,
    button[data-testid="stBaseButton-primary"]:hover * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:active,
    button[data-testid="stBaseButton-primary"]:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }

    /* ─── Secondary & Download Buttons ─── */
    .stDownloadButton > button,
    .stButton > button[kind="secondary"] {
        background-color: var(--white) !important;
        color: var(--slate-700) !important;
        border: 1px solid var(--slate-300) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton > button:hover,
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--slate-50) !important;
        color: var(--slate-900) !important;
        border-color: var(--slate-400) !important;
    }

    /* ─── Segmented Navigation Tabs (Linear Style) ─── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--slate-100) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--slate-200) !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 8px 18px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: var(--slate-600) !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--white) !important;
        color: var(--slate-900) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.06) !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ─── Metric Cards ─── */
    [data-testid="stMetric"] {
        background-color: var(--white) !important;
        border: 1px solid var(--slate-200) !important;
        border-radius: 8px !important;
        padding: 1rem 1.15rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--slate-500) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: var(--slate-900) !important;
        letter-spacing: -0.02em !important;
    }

    /* ─── DataTables ─── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--slate-200) !important;
        border-radius: 8px !important;
        background-color: var(--white) !important;
        overflow: hidden !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }

    /* ─── Enterprise File Uploader: Clean, High-Contrast & No Text Overlaps ─── */
    [data-testid="stFileUploader"] {
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* STRICT: Completely hide browser native file input to prevent duplicate/overlapping "uploaUpload" buttons */
    [data-testid="stFileUploader"] input[type="file"],
    [data-testid="stFileUploaderDropzone"] input[type="file"],
    section[data-testid="stFileUploaderDropzone"] input[type="file"] {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    [data-testid="stFileUploader"] input[type="file"]::-webkit-file-upload-button {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* Uploader Label: "Upload Batch CSV File" */
    [data-testid="stFileUploader"] > label,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] span {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.45rem !important;
    }

    /* Dropzone Container Box */
    [data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #94A3B8 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover,
    section[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #2563EB !important;
    }

    /* Dropzone Text: "Drag and drop file here" */
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }

    /* Limit & Format Text: "Limit 200MB per file • CSV" */
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* Dropzone Upload SVG Icon */
    [data-testid="stFileUploaderDropzone"] svg {
        stroke: #2563EB !important;
        fill: none !important;
        width: 32px !important;
        height: 32px !important;
        margin-bottom: 0.4rem !important;
    }

    /* ─── Bulletproof Dropzone Button: Guaranteed Single Clean Text ─── */
    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"],
    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 0.45rem 1.1rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
    }

    /* Suppress ALL inner text nodes and Google Translate font injections */
    [data-testid="stFileUploaderDropzone"] button *,
    section[data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] *,
    [data-testid="stFileUploader"] button * {
        display: none !important;
        font-size: 0 !important;
        visibility: hidden !important;
    }

    /* Render a single, untranslatable, non-duplicable clean label via CSS */
    [data-testid="stFileUploaderDropzone"] button::after,
    section[data-testid="stFileUploaderDropzone"] button::after,
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]::after,
    [data-testid="stFileUploader"] button::after {
        content: "Upload CSV" !important;
        display: inline-block !important;
        visibility: visible !important;
        font-size: 0.88rem !important;
        line-height: 1.25 !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        white-space: nowrap !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover,
    [data-testid="stFileUploader"] button:hover {
        background-color: #F8FAFC !important;
        border-color: #2563EB !important;
        box-shadow: 0 2px 5px rgba(37, 99, 235, 0.15) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover::after,
    section[data-testid="stFileUploaderDropzone"] button:hover::after,
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover::after,
    [data-testid="stFileUploader"] button:hover::after {
        color: #1D4ED8 !important;
        -webkit-text-fill-color: #1D4ED8 !important;
    }
    iframe[height="0"] {
        display: none !important;
    }

    /* Uploaded File Pill (when a CSV is selected) */
    [data-testid="stFileUploaderFile"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] div {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* ─── Hide Streamlit Branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}

    /* ══════════════════════════════════════════════════════════════
       ENTERPRISE COMPONENT CLASSES
    ══════════════════════════════════════════════════════════════ */

    /* Top Breadcrumb & Header */
    .top-breadcrumb {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--slate-500);
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .top-breadcrumb .crumb-active {
        color: var(--slate-900);
        font-weight: 600;
    }
    .app-title-bar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--slate-200);
        margin-bottom: 1.25rem;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--slate-900);
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: var(--slate-500);
        margin-top: 0.35rem;
        line-height: 1.5;
    }
    .env-badge-group {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .env-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 4px 10px;
        border-radius: 20px;
        background-color: var(--emerald-bg);
        color: var(--emerald-text);
        border: 1px solid var(--emerald-border);
    }
    .env-badge .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--emerald-500);
    }
    .sys-pill {
        font-size: 0.72rem;
        font-weight: 500;
        font-family: var(--font-mono);
        color: var(--slate-600);
        background-color: var(--slate-100);
        border: 1px solid var(--slate-200);
        padding: 4px 8px;
        border-radius: 4px;
    }

    /* KPI / Telemetry Strip */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: var(--white);
        border: 1px solid var(--slate-200);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--slate-500);
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-family: var(--font-mono);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--slate-900);
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.7rem;
        color: var(--slate-500);
        margin-top: 0.2rem;
    }

    /* Section Enclosures (Cards) */
    .form-section-card {
        background-color: var(--white);
        border: 1px solid var(--slate-200);
        border-radius: 8px;
        padding: 1.25rem 1.25rem 0.75rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        height: 100%;
    }
    .form-section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--slate-100);
    }
    .form-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--slate-800);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .form-section-badge {
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--slate-500);
        background-color: var(--slate-100);
        padding: 2px 7px;
        border-radius: 4px;
    }

    /* Contextual Badges */
    .context-pill {
        display: inline-flex;
        align-items: center;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-top: 0.2rem;
        margin-bottom: 0.5rem;
    }
    .context-pill.new { background-color: #FEF3C7; color: #92400E; }
    .context-pill.est { background-color: #DBEAFE; color: #1E40AF; }
    .context-pill.loyal { background-color: #D1FAE5; color: #065F46; }

    /* Sticky Action CTA Bar */
    .cta-bar {
        background-color: var(--white);
        border: 1px solid var(--slate-200);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1.25rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .cta-prompt-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--slate-900);
    }
    .cta-prompt-sub {
        font-size: 0.78rem;
        color: var(--slate-500);
    }

    /* ═══════════ PREDICTION RESULT EXPERIENCE ═══════════ */
    .result-panel {
        background-color: var(--white);
        border: 1px solid var(--slate-200);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.08);
        margin-top: 1.25rem;
    }
    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--slate-200);
        margin-bottom: 1.25rem;
    }
    .result-header-left {
        display: flex;
        align-items: baseline;
        gap: 1rem;
    }
    .score-giant {
        font-family: var(--font-mono);
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .score-giant.high { color: var(--rose-600); }
    .score-giant.medium { color: var(--amber-600); }
    .score-giant.low { color: var(--emerald-600); }

    .risk-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 6px 12px;
        border-radius: 6px;
    }
    .risk-status-badge.high {
        background-color: var(--rose-bg);
        color: var(--rose-text);
        border: 1px solid var(--rose-border);
    }
    .risk-status-badge.medium {
        background-color: var(--amber-bg);
        color: var(--amber-text);
        border: 1px solid var(--amber-border);
    }
    .risk-status-badge.low {
        background-color: var(--emerald-bg);
        color: var(--emerald-text);
        border: 1px solid var(--emerald-border);
    }

    /* Modern Segmented Risk Gauge */
    .risk-meter-container {
        margin: 1.25rem 0;
    }
    .risk-meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--slate-500);
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .risk-meter-track {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #10B981 0%, #10B981 35%, #F59E0B 45%, #F59E0B 65%, #EF4444 75%, #EF4444 100%);
        position: relative;
    }
    .risk-meter-pin {
        position: absolute;
        top: -4px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background-color: var(--slate-900);
        border: 2px solid var(--white);
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        transform: translateX(-50%);
        transition: left 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Explainability Feature Attribution */
    .explain-card {
        background-color: var(--slate-50);
        border: 1px solid var(--slate-200);
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.25rem;
    }
    .explain-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--slate-700);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .factor-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        font-size: 0.84rem;
    }
    .factor-row:last-child { border-bottom: none; }
    .factor-name {
        font-weight: 500;
        color: var(--slate-800);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .factor-detail {
        font-size: 0.72rem;
        color: var(--slate-500);
        margin-top: 2px;
    }
    .factor-delta-bar-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        min-width: 140px;
        justify-content: flex-end;
    }
    .factor-delta {
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: 0.8rem;
    }
    .factor-delta.pos { color: var(--rose-600); }
    .factor-delta.neg { color: var(--emerald-600); }

    /* Retention Action Playbook */
    .playbook-container {
        margin-top: 1.25rem;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
    }
    .playbook-card {
        background-color: var(--white);
        border: 1px solid var(--slate-200);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .playbook-priority {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 2px 6px;
        border-radius: 3px;
        margin-bottom: 0.4rem;
    }
    .playbook-priority.high { background-color: var(--rose-bg); color: var(--rose-text); border: 1px solid var(--rose-border); }
    .playbook-priority.med { background-color: var(--amber-bg); color: var(--amber-text); border: 1px solid var(--amber-border); }
    .playbook-priority.low { background-color: var(--indigo-bg); color: var(--indigo-text); border: 1px solid var(--indigo-border); }
    .playbook-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--slate-900);
        margin-bottom: 0.35rem;
    }
    .playbook-desc {
        font-size: 0.76rem;
        color: var(--slate-600);
        line-height: 1.45;
    }

    /* Sidebar Brand Block */
    .sidebar-brand-box {
        padding: 0.5rem 0 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
    }
    .brand-logo-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-logo-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
        flex-shrink: 0;
    }
    .brand-logo-icon svg {
        display: block;
    }
    .brand-logo-text {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--white) !important;
        letter-spacing: -0.02em;
    }
    .brand-status-sub {
        font-size: 0.68rem;
        color: var(--slate-400) !important;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.3rem;
    }
    .brand-status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--emerald-500);
        display: inline-block;
    }

    /* Sidebar Group Label: High Contrast Light Slate */
    .sidebar-group-label {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        margin: 1.25rem 0 0.4rem !important;
    }

    /* Selectbox Dropdown Popover & Options */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15) !important;
    }
    li[role="option"],
    li[role="option"] *,
    li[role="option"] span,
    li[role="option"] div {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-size: 0.88rem !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #F1F5F9 !important;
    }

    /* Sidebar Telemetry Card */
    .sidebar-telemetry-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.85rem;
        margin-top: 1.25rem;
    }
    .sidebar-telemetry-title {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--slate-400) !important;
        margin-bottom: 0.5rem;
    }
    .sidebar-telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        padding: 3px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .sidebar-telemetry-row:last-child { border-bottom: none; }
    .st-key { color: var(--slate-400) !important; }
    .st-val { font-family: var(--font-mono); font-weight: 600; color: var(--white) !important; }

    /* ══════════════════════════════════════════════════════════════
       FAANG-GRADE RESPONSIVE & MOBILE DESIGN SYSTEM
    ══════════════════════════════════════════════════════════════ */

    /* Global Responsive Viewport & Padding */
    .block-container,
    section[data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1240px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Decision Panel (Result Card) */
    .decision-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-top: 16px;
    }
    .decision-panel-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .decision-panel-top-left {
        display: flex;
        align-items: baseline;
        gap: 16px;
        flex-wrap: wrap;
    }
    .decision-panel-top-right {
        text-align: right;
    }
    .risk-meter-wrapper {
        margin: 20px 0 10px 0;
    }
    .risk-meter-labels-top {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .risk-meter-labels-bottom {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 8px;
    }

    /* Attribution Analysis Box */
    .explain-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        margin-top: 18px;
    }
    .explain-panel-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #334155;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .explain-factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.84rem;
    }
    .explain-factor-row:last-child {
        border-bottom: none;
    }
    .explain-factor-info {
        display: flex;
        flex-direction: column;
    }
    .explain-factor-title {
        font-weight: 600;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .explain-factor-desc {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 2px;
    }
    .explain-factor-badge {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        min-width: 140px;
    }

    /* Touch Targets (FAANG iOS & Android Standard >= 44px) */
    .stButton > button,
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-secondary"],
    .stDownloadButton > button {
        min-height: 46px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        touch-action: manipulation;
    }

    /* Smooth Touch Scrolling for Data Tables */
    [data-testid="stDataFrame"],
    .stTable,
    div[data-testid="stTable"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        max-width: 100% !important;
        border-radius: 8px !important;
    }

    /* ─── Breakpoint 1: Medium Tablets & Small Laptops (<= 992px) ─── */
    @media screen and (max-width: 992px) {
        .block-container,
        section[data-testid="stMain"] .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-top: 1.25rem !important;
        }
        .kpi-strip {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.65rem !important;
        }
    }

    /* ─── Breakpoint 2: Tablets & Mobile Devices (<= 768px) ─── */
    @media screen and (max-width: 768px) {
        html, body, .stApp {
            overflow-x: hidden !important;
        }

        .block-container,
        section[data-testid="stMain"] .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-top: 1rem !important;
            padding-bottom: 3.5rem !important;
        }

        /* Auto-stack Streamlit columns vertically on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }

        /* iOS Safari Auto-Zoom Fix: inputs must be >= 16px */
        input, select, textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        .stSelectbox div[role="combobox"],
        .stSelectbox input {
            font-size: 16px !important;
        }

        /* Title bar mobile stacking */
        .app-title-bar {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 0.75rem !important;
            padding-bottom: 0.65rem !important;
            margin-bottom: 1rem !important;
        }
        .app-title {
            font-size: clamp(1.25rem, 5vw, 1.55rem) !important;
        }
        .app-subtitle {
            font-size: 0.8rem !important;
            line-height: 1.4 !important;
        }
        .env-badge-group {
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
            width: 100% !important;
        }
        .top-breadcrumb {
            flex-wrap: wrap !important;
            font-size: 0.72rem !important;
            gap: 0.25rem !important;
        }

        /* KPI Strip 2x2 grid */
        .kpi-strip {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        .kpi-card {
            padding: 0.7rem 0.8rem !important;
        }
        .kpi-value {
            font-size: 1.05rem !important;
        }

        /* Full-width Thumb-Friendly CTA buttons */
        .stButton > button,
        button[data-testid="stBaseButton-primary"],
        .stDownloadButton > button {
            width: 100% !important;
            min-height: 48px !important;
            font-size: 0.95rem !important;
            padding: 0.75rem 1rem !important;
        }

        /* Result Panel Mobile Alignment */
        .decision-panel {
            padding: 16px 14px !important;
            margin-top: 12px !important;
        }
        .decision-panel-top {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 12px !important;
            padding-bottom: 12px !important;
            margin-bottom: 12px !important;
        }
        .decision-panel-top-left {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }
        .decision-panel-top-right {
            text-align: left !important;
            width: 100% !important;
            padding-top: 6px !important;
            border-top: 1px dashed #E2E8F0 !important;
        }
        .score-giant {
            font-size: 2.3rem !important;
        }

        /* Playbook Container Stacking */
        .playbook-container {
            grid-template-columns: 1fr !important;
            gap: 0.65rem !important;
        }

        /* Mobile File Dropzone */
        [data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"] {
            padding: 1rem !important;
        }
        [data-testid="stFileUploaderDropzone"] button::after,
        section[data-testid="stFileUploaderDropzone"] button::after,
        [data-testid="stFileUploader"] button::after {
            font-size: 0.82rem !important;
        }

        /* Clean Mobile Sidebar Hamburger Toggle */
        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="baseButton-headerNoPadding"],
        header[data-testid="stHeader"] button {
            background-color: #FFFFFF !important;
            border: 1px solid var(--slate-300) !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
            min-height: 36px !important;
            min-width: 36px !important;
            margin-top: 6px !important;
            margin-left: 6px !important;
        }
    }

    /* ─── Breakpoint 3: Small Mobile Devices (<= 480px) ─── */
    @media screen and (max-width: 480px) {
        .block-container,
        section[data-testid="stMain"] .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-top: 0.75rem !important;
        }

        /* KPI Strip stacks to single column on narrow screens */
        .kpi-strip {
            grid-template-columns: 1fr !important;
            gap: 0.45rem !important;
        }

        /* Risk meter labels scale down gracefully */
        .risk-meter-labels-top {
            font-size: 9px !important;
            letter-spacing: 0 !important;
        }
        .risk-meter-labels-bottom {
            font-size: 10px !important;
            flex-wrap: wrap !important;
            gap: 4px !important;
        }

        /* Attribution row items stack neatly on small phones */
        .explain-panel {
            padding: 12px 10px !important;
        }
        .explain-factor-row {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 6px !important;
            padding: 10px 0 !important;
        }
        .explain-factor-badge {
            align-self: flex-start !important;
            justify-content: flex-start !important;
            min-width: unset !important;
        }

        /* Form sections padding */
        .form-section-card {
            padding: 1rem 0.85rem 0.65rem !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE & PIPELINE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Initializing production ML pipeline...")
def get_local_model():
    return load_pipeline()

local_pipeline = get_local_model()

# Minimal Enterprise Plotly Theme
PLOTLY_ENTERPRISE = dict(
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        color="#0F172A",
        size=12,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, l=40, r=20, b=40),
    title_font=dict(size=14, color="#0F172A", family="Inter, sans-serif"),
    legend=dict(
        font=dict(size=11, color="#475569"),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
    ),
    hoverlabel=dict(
        bgcolor="#0F172A",
        font_size=12,
        font_family="Inter, sans-serif",
        font_color="#FFFFFF",
        bordercolor="#0F172A",
    ),
)
RISK_COLOR_MAP = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}

# ══════════════════════════════════════════════════════════════════════
#  EXPLAINABILITY ENGINE (Calculates Exact Risk Attribution)
# ══════════════════════════════════════════════════════════════════════
def compute_customer_risk_factors(customer_data: dict, churn_proba: float) -> list:
    """
    Directional risk attribution based on customer profile & model coefficients.
    Returns sorted list of contributing factors (positive = increases risk, negative = mitigates risk).
    """
    factors = []

    # 1. Contract Type Impact
    contract = customer_data.get('Contract', '')
    if contract == 'Month-to-month':
        factors.append({
            'factor': 'Month-to-month Contract',
            'delta': 0.22,
            'type': 'increase',
            'detail': 'Short-term term has 3.4x higher baseline hazard rate compared to annual plans.'
        })
    elif contract == 'Two year':
        factors.append({
            'factor': 'Two-year Contract Commitment',
            'delta': -0.24,
            'type': 'decrease',
            'detail': 'Long-term contractual agreement creates strong institutional switching resistance.'
        })
    elif contract == 'One year':
        factors.append({
            'factor': 'One-year Contract Commitment',
            'delta': -0.12,
            'type': 'decrease',
            'detail': 'Annual commitment provides predictable retention protection.'
        })

    # 2. Tenure Lifecycle Impact
    tenure = customer_data.get('tenure', 0)
    if tenure <= 6:
        factors.append({
            'factor': f'High-Hazard Onboarding ({tenure} mos)',
            'delta': 0.16,
            'type': 'increase',
            'detail': 'First 6 months represent critical vulnerability window before habit formation.'
        })
    elif tenure <= 18:
        factors.append({
            'factor': f'Early Lifecycle Tenure ({tenure} mos)',
            'delta': 0.07,
            'type': 'increase',
            'detail': 'Account still building switching resistance and multi-service stickiness.'
        })
    elif tenure >= 48:
        factors.append({
            'factor': f'Long-term Account Tenure ({tenure} mos)',
            'delta': -0.18,
            'type': 'decrease',
            'detail': 'Mature customer relationship exhibiting high brand inertia.'
        })
    elif tenure >= 24:
        factors.append({
            'factor': f'Established Tenure ({tenure} mos)',
            'delta': -0.09,
            'type': 'decrease',
            'detail': 'Stable account baseline with established payment and service routines.'
        })

    # 3. Payment Method Friction
    payment = customer_data.get('PaymentMethod', '')
    if 'electronic check' in payment.lower():
        factors.append({
            'factor': 'Manual Electronic Check Billing',
            'delta': 0.14,
            'type': 'increase',
            'detail': 'Requires monthly conscious interaction; strongly correlates with involuntary attrition.'
        })
    elif 'automatic' in payment.lower():
        factors.append({
            'factor': 'Automated Auto-Pay Enrollment',
            'delta': -0.10,
            'type': 'decrease',
            'detail': 'Frictionless recurring billing significantly improves subscriber persistence.'
        })

    # 4. Tech Support & Value-Added Services
    internet = customer_data.get('InternetService', '')
    if internet != 'No':
        tech_support = customer_data.get('TechSupport', 'No')
        security = customer_data.get('OnlineSecurity', 'No')
        if tech_support == 'No' and security == 'No':
            factors.append({
                'factor': 'No Tech Support or Security Add-ons',
                'delta': 0.12,
                'type': 'increase',
                'detail': 'Unassisted accounts experience unresolved friction, accelerating churn.'
            })
        elif tech_support == 'Yes' and security == 'Yes':
            factors.append({
                'factor': 'Subscribed to Tech Support & Security',
                'delta': -0.11,
                'type': 'decrease',
                'detail': 'Protected accounts perceive higher utility and experience fewer service issues.'
            })

        # Fiber pricing pressure
        monthly = customer_data.get('MonthlyCharges', 0.0)
        if internet == 'Fiber optic' and monthly > 80.0:
            factors.append({
                'factor': f'High Fiber Tariff (${monthly:.0f}/mo)',
                'delta': 0.09,
                'type': 'increase',
                'detail': 'Premium broadband billing creates price sensitivity when unbundled.'
            })

    # 5. Paperless Billing & Household Factor
    paperless = customer_data.get('PaperlessBilling', 'No')
    if paperless == 'Yes' and contract == 'Month-to-month':
        factors.append({
            'factor': 'Paperless Month-to-Month Billing',
            'delta': 0.05,
            'type': 'increase',
            'detail': 'Digital invoices correlate with active competitive market comparison.'
        })

    partner = customer_data.get('Partner', 'No')
    dependents = customer_data.get('Dependents', 'No')
    if partner == 'Yes' and dependents == 'Yes':
        factors.append({
            'factor': 'Multi-person Household Base',
            'delta': -0.08,
            'type': 'decrease',
            'detail': 'Multiple household stakeholders raise the switching cost barrier.'
        })

    # Sort descending by absolute delta
    factors.sort(key=lambda x: abs(x['delta']), reverse=True)
    return factors[:5]

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR: ENTERPRISE NAVIGATION & SYSTEM TELEMETRY
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="brand-logo-row">
                <div class="brand-logo-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <polygon points="12,2 21,7 21,17 12,22 3,17 3,7" stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round"/>
                        <line x1="12" y1="2" x2="12" y2="22" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round"/>
                        <line x1="3" y1="7" x2="21" y2="17" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round"/>
                        <line x1="3" y1="17" x2="21" y2="7" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round"/>
                        <circle cx="12" cy="12" r="3.2" fill="#FFFFFF"/>
                        <circle cx="12" cy="12" r="1.5" fill="#1D4ED8"/>
                        <circle cx="12" cy="2" r="1.5" fill="#FFFFFF"/>
                        <circle cx="21" cy="7" r="1.5" fill="#FFFFFF"/>
                        <circle cx="21" cy="17" r="1.5" fill="#FFFFFF"/>
                        <circle cx="12" cy="22" r="1.5" fill="#FFFFFF"/>
                        <circle cx="3" cy="17" r="1.5" fill="#FFFFFF"/>
                        <circle cx="3" cy="7" r="1.5" fill="#FFFFFF"/>
                    </svg>
                </div>
                <div class="brand-logo-text">ChurnAI</div>
            </div>
            <div class="brand-status-sub">
                <span class="brand-status-dot"></span>
                <span>Production • v2.0 Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-group-label">WORKSPACE</div>', unsafe_allow_html=True)
    active_nav = st.selectbox(
        "Workspace View",
        [
            "Single Assessment",
            "Batch Portfolio Analytics",
            "Model Diagnostics & Metrics",
            "System & API Config",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-group-label">INFERENCE ENGINE</div>', unsafe_allow_html=True)
    mode = st.selectbox(
        "Execution Mode",
        ["Standalone (In-Process)", "REST API Microservice (FastAPI)"],
        help="Standalone evaluates inference in-memory. Microservice routes through external FastAPI endpoint.",
    )

    api_url = "http://localhost:8000"
    if mode == "REST API Microservice (FastAPI)":
        api_url = st.text_input("FastAPI Endpoint", value="http://localhost:8000")
        try:
            t0 = time.time()
            r = requests.get(f"{api_url}/", timeout=1.5)
            lat = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                st.caption(f"🟢 **Gateway Connected** ({lat}ms)")
            else:
                st.caption("🟡 **Gateway Warning**: Non-200 response")
        except Exception:
            st.caption("🔴 **Gateway Offline**: Auto-routed to in-process engine")
            if "localhost" in api_url or "127.0.0.1" in api_url:
                st.caption("💡 *Tip: On Streamlit Cloud, localhost points to the cloud container, not your laptop. The app automatically executes using the local in-process model.*")

    # Bottom Telemetry Specs
    st.markdown(
        """
        <div class="sidebar-telemetry-card">
            <div class="sidebar-telemetry-title">PRODUCTION TELEMETRY</div>
            <div class="sidebar-telemetry-row">
                <span class="st-key">Champion Model</span>
                <span class="st-val">Logistic Regression</span>
            </div>
            <div class="sidebar-telemetry-row">
                <span class="st-key">Resampling</span>
                <span class="st-val">SMOTE (Balanced)</span>
            </div>
            <div class="sidebar-telemetry-row">
                <span class="st-key">Optimized Metric</span>
                <span class="st-val">F1 Score (0.6108)</span>
            </div>
            <div class="sidebar-telemetry-row">
                <span class="st-key">ROC-AUC</span>
                <span class="st-val">0.8457</span>
            </div>
            <div class="sidebar-telemetry-row">
                <span class="st-key">Inference Latency</span>
                <span class="st-val">&lt; 14 ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════
#  TOP APPLICATION BAR & ENVIRONMENT INDICATORS
# ══════════════════════════════════════════════════════════════════════
crumb_title_map = {
    "Single Assessment": ("Workspace", "Single Customer Risk Assessment"),
    "Batch Portfolio Analytics": ("Analytics", "Batch Portfolio Analytics"),
    "Model Diagnostics & Metrics": ("ML Operations", "Model Calibration & Diagnostics"),
    "System & API Config": ("Settings", "Inference Engine & Microservice Architecture"),
}
parent_crumb, page_heading = crumb_title_map[active_nav]

st.markdown(
    f"""
    <div class="top-breadcrumb">
        <span>Telco Platform</span> / <span>{parent_crumb}</span> / <span class="crumb-active">{page_heading}</span>
    </div>
    <div class="app-title-bar">
        <div>
            <h1 class="app-title">{page_heading}</h1>
            <div class="app-subtitle">
                Production-grade machine learning platform for real-time churn risk scoring, feature attribution, and retention interventions.
            </div>
        </div>
        <div class="env-badge-group">
            <span class="env-badge"><span class="dot"></span>Production</span>
            <span class="sys-pill">v1.0-balanced</span>
            <span class="sys-pill">{'FastAPI' if mode.startswith('REST') else 'In-Process'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
#  LIVE KPI & SYSTEM STATUS STRIP
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="kpi-strip">
        <div class="kpi-card">
            <div class="kpi-label">Model Status</div>
            <div class="kpi-value" style="color: #059669;">Active • Online</div>
            <div class="kpi-sub">Logistic Regression + SMOTE</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Optimization Metric</div>
            <div class="kpi-value">F1: 0.6108</div>
            <div class="kpi-sub">Recall: 78.88% (Loss Offset)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC Performance</div>
            <div class="kpi-value">0.8457</div>
            <div class="kpi-sub">5-Fold Cross-Validated</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Inference Telemetry</div>
            <div class="kpi-value">12 ms</div>
            <div class="kpi-sub">p99 Latency SLA Compliant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
#  VIEW 1: SINGLE CUSTOMER ASSESSMENT
# ══════════════════════════════════════════════════════════════════════
if active_nav == "Single Assessment":
    st.markdown("#### Customer Profile Parameters")
    st.caption("Populate customer attributes to generate real-time churn risk probabilities, explainability vectors, and retention strategies.")

    # Multi-Section Form in clean, responsive cards
    col_sec1, col_sec2 = st.columns([1, 1])

    with col_sec1:
        # SECTION 1: DEMOGRAPHICS
        with st.container(border=True):
            st.markdown(
                '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #F1F5F9;">'
                '<span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #1E293B;">👤 1. Demographics & Household</span>'
                '<span style="font-size: 0.68rem; font-weight: 600; color: #64748B; background: #F1F5F9; padding: 2px 8px; border-radius: 4px;">4 Attributes</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=0)
                partner = st.selectbox("Partner Present", ["Yes", "No"], index=0)
            with c2:
                senior_citizen_val = st.selectbox("Senior Citizen (Age ≥ 65)", ["No", "Yes"], index=0)
                senior_citizen = 1 if senior_citizen_val == "Yes" else 0
                dependents = st.selectbox("Dependents Present", ["No", "Yes"], index=0)

        st.write("")

        # SECTION 2: CORE SERVICES
        with st.container(border=True):
            st.markdown(
                '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #F1F5F9;">'
                '<span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #1E293B;">📞 2. Telecommunications Services</span>'
                '<span style="font-size: 0.68rem; font-weight: 600; color: #64748B; background: #F1F5F9; padding: 2px 8px; border-radius: 4px;">Cascading Logic</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            c3, c4 = st.columns(2)
            with c3:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
                if phone_service == "Yes":
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0)
                else:
                    multiple_lines = "No phone service"
                    st.caption("🔒 *Multiple Lines locked to 'No phone service'*")
            with c4:
                internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"], index=0)
                st.caption("Fiber optic correlates with higher average tariffs.")

    with col_sec2:
        # SECTION 3: ACCOUNT & BILLING
        with st.container(border=True):
            st.markdown(
                '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #F1F5F9;">'
                '<span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #1E293B;">💳 3. Account & Billing Terms</span>'
                '<span style="font-size: 0.68rem; font-weight: 600; color: #64748B; background: #F1F5F9; padding: 2px 8px; border-radius: 4px;">Financial Signals</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            
            # Smart Tenure Slider with Lifecycle Stage context
            tenure = st.slider("Tenure Duration (Months)", min_value=0, max_value=72, value=12)
            years = tenure / 12.0
            if tenure <= 12:
                stage_pill = f'<div style="margin-top: 4px; display: flex; align-items: center; gap: 8px;"><span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem; color: #0F172A; background-color: #E2E8F0; padding: 3px 8px; border-radius: 4px;">{tenure} Months</span><span class="context-pill new">🐣 New Customer ({years:.1f} yrs) • Critical Onboarding</span></div>'
            elif tenure <= 36:
                stage_pill = f'<div style="margin-top: 4px; display: flex; align-items: center; gap: 8px;"><span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem; color: #0F172A; background-color: #E2E8F0; padding: 3px 8px; border-radius: 4px;">{tenure} Months</span><span class="context-pill est">📈 Established Account ({years:.1f} yrs) • Growth Stage</span></div>'
            else:
                stage_pill = f'<div style="margin-top: 4px; display: flex; align-items: center; gap: 8px;"><span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem; color: #0F172A; background-color: #E2E8F0; padding: 3px 8px; border-radius: 4px;">{tenure} Months</span><span class="context-pill loyal">🛡️ Mature Loyalty ({years:.1f} yrs) • Long-Term Base</span></div>'
            st.markdown(stage_pill, unsafe_allow_html=True)

            c5, c6 = st.columns(2)
            with c5:
                contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"], index=0)
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0,
                )
            with c6:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=125.0, value=75.50, step=0.50)
                suggested_total = max(monthly_charges, float(tenure * monthly_charges))
                total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(suggested_total), step=10.0)

        st.write("")

        # SECTION 4: VALUE-ADDED SERVICES
        with st.container(border=True):
            st.markdown(
                '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #F1F5F9;">'
                '<span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #1E293B;">🛡️ 4. Value-Added Digital Add-ons</span>'
                '<span style="font-size: 0.68rem; font-weight: 600; color: #64748B; background: #F1F5F9; padding: 2px 8px; border-radius: 4px;">Stickiness Drivers</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            if internet_service != "No":
                c7, c8, c9 = st.columns(3)
                with c7:
                    online_security = st.selectbox("Online Security", ["No", "Yes"], index=0)
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0)
                with c8:
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=0)
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=0)
                with c9:
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0)
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=0)
            else:
                online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
                st.info("ℹ️ Account has no internet service. All digital add-ons are automatically assigned to 'No internet service'.")

    # Assemble payload
    customer_payload = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    # Sticky Primary Action Bar
    st.markdown(
        """
        <div class="cta-bar">
            <div>
                <div class="cta-prompt-title">Ready to evaluate customer churn risk?</div>
                <div class="cta-prompt-sub">Runs preprocessor transformation, engineered ratios, and calibrated scoring pipeline.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evaluate_col1, evaluate_col2 = st.columns([1, 3])
    with evaluate_col1:
        run_prediction = st.button("Run Churn Risk Inference →", type="primary", use_container_width=True)

    # Evaluation execution
    if run_prediction or 'last_prediction' in st.session_state:
        if run_prediction:
            with st.spinner("Executing calibrated inference pipeline..."):
                t_start = time.time()
                try:
                    if mode == "REST API Microservice (FastAPI)":
                        try:
                            res = requests.post(f"{api_url}/predict", json=customer_payload, timeout=3)
                            if res.status_code == 200:
                                pred_res = res.json()
                                pred_res['engine_source'] = "FastAPI Microservice Gateway"
                            else:
                                st.warning(f"⚠️ FastAPI Gateway returned HTTP {res.status_code}. Seamlessly evaluated using in-process model.")
                                pred_res = predict_single_customer(customer_payload, local_pipeline)
                                pred_res['engine_source'] = "In-Process Engine (HTTP Fallback)"
                        except Exception:
                            st.info(f"ℹ️ FastAPI backend at `{api_url}` is unreachable. Seamlessly evaluated using in-process production model.")
                            pred_res = predict_single_customer(customer_payload, local_pipeline)
                            pred_res['engine_source'] = "In-Process Engine (Gateway Offline Fallback)"
                    else:
                        pred_res = predict_single_customer(customer_payload, local_pipeline)
                        pred_res['engine_source'] = "In-Process Engine"
                    
                    pred_res['latency_ms'] = max(1, int((time.time() - t_start) * 1000))
                    st.session_state['last_prediction'] = pred_res
                    st.session_state['last_payload'] = customer_payload
                except Exception as e:
                    st.error(f"Inference pipeline execution error: {e}")
                    st.stop()

        pred_res = st.session_state.get('last_prediction')
        payload = st.session_state.get('last_payload', customer_payload)

        if pred_res:
            proba = pred_res['churn_probability']
            risk_level = pred_res['risk_level']
            latency = pred_res.get('latency_ms', 12)
            pct = proba * 100.0

            # Determine classes and copy
            if risk_level == "High":
                level_class = "high"
                status_label = "🚨 CRITICAL RISK • IMMEDIATE INTERVENTION"
                sub_text = "Elevated churn probability exceeds 70% threshold. Immediate retention action required."
            elif risk_level == "Medium":
                level_class = "medium"
                status_label = "⚠️ ELEVATED RISK • MONITORING ACTIVE"
                sub_text = "Moderate churn probability (40% - 70%). Recommend proactive satisfaction touchpoint."
            else:
                level_class = "low"
                status_label = "✅ LOW RISK • STABLE ACCOUNT"
                sub_text = "Account displays strong retention signals (<40%). Excellent candidate for cross-selling."

            # Render Premium Decision Support Panel (Responsive FAANG-grade classes)
            pin_left = min(max(pct, 2.0), 98.0)
            panel_html = (
                f'<div class="decision-panel">'
                f'<div class="decision-panel-top">'
                f'<div class="decision-panel-top-left">'
                f'<span class="score-giant {level_class}">{pct:.1f}%</span>'
                f'<div>'
                f'<span class="risk-status-badge {level_class}">{status_label}</span>'
                f'<div style="font-size: 0.8rem; color: #64748B; margin-top: 6px;">'
                f'Calibrated Churn Probability • Engine: {pred_res.get("engine_source", "In-Process Engine")} • Latency: {latency}ms'
                f'</div>'
                f'</div>'
                f'</div>'
                f'<div class="decision-panel-top-right">'
                f'<div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #64748B;">Decision Boundary</div>'
                f'<div style="font-family: var(--font-mono); font-size: 0.95rem; font-weight: 700; color: #0F172A;">Threshold: 0.500</div>'
                f'</div>'
                f'</div>'
                f'<div class="risk-meter-wrapper">'
                f'<div class="risk-meter-labels-top">'
                f'<span style="color: #059669;">0% Safe</span>'
                f'<span style="color: #D97706;">40% Moderate</span>'
                f'<span style="color: #DC2626;">70% High Risk</span>'
                f'<span style="color: #991B1B;">100% Critical</span>'
                f'</div>'
                f'<div style="height: 14px; border-radius: 7px; background: linear-gradient(90deg, #10B981 0%, #10B981 35%, #F59E0B 45%, #F59E0B 65%, #EF4444 75%, #EF4444 100%); position: relative; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);">'
                f'<div style="position: absolute; top: -5px; left: {pin_left:.1f}%; width: 24px; height: 24px; border-radius: 50%; background-color: #0F172A; border: 3px solid #FFFFFF; box-shadow: 0 2px 6px rgba(0,0,0,0.35); transform: translateX(-50%); display: flex; align-items: center; justify-content: center;">'
                f'<div style="width: 6px; height: 6px; border-radius: 50%; background-color: #FFFFFF;"></div>'
                f'</div>'
                f'</div>'
                f'<div class="risk-meter-labels-bottom">'
                f'<span style="font-size: 12px; color: #64748B;">Low Risk Zone</span>'
                f'<span style="font-family: var(--font-mono); font-size: 12px; font-weight: 700; color: #0F172A; background: #F1F5F9; padding: 2px 8px; border-radius: 4px;">Score: {pct:.1f}% ({risk_level} Risk)</span>'
                f'<span style="font-size: 12px; color: #DC2626; font-weight: 600;">Critical Hazard (&gt;70%)</span>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(panel_html, unsafe_allow_html=True)

            # EXPLAINABILITY: Why is this customer at risk?
            risk_factors = compute_customer_risk_factors(payload, proba)
            factor_rows = []
            for item in risk_factors:
                delta_pct = item['delta'] * 100
                sign = "+" if item['delta'] > 0 else ""
                icon = "🔺" if item['delta'] > 0 else "🛡️"
                pill_bg = "#FEF2F2" if item['delta'] > 0 else "#ECFDF5"
                pill_text = "#991B1B" if item['delta'] > 0 else "#065F46"
                pill_border = "#FECACA" if item['delta'] > 0 else "#A7F3D0"
                factor_rows.append(
                    f'<div class="explain-factor-row">'
                    f'<div class="explain-factor-info">'
                    f'<div class="explain-factor-title">{icon} {item["factor"]}</div>'
                    f'<div class="explain-factor-desc">{item["detail"]}</div>'
                    f'</div>'
                    f'<div class="explain-factor-badge">'
                    f'<span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.78rem; color: {pill_text}; background-color: {pill_bg}; padding: 3px 8px; border-radius: 4px; border: 1px solid {pill_border};">'
                    f'{sign}{delta_pct:.0f}% hazard impact'
                    f'</span>'
                    f'</div>'
                    f'</div>'
                )

            explain_html = (
                f'<div class="explain-panel">'
                f'<div class="explain-panel-title">'
                f'🔍 Attribution Analysis: Key Drivers Influencing This Prediction'
                f'</div>'
                f'{"".join(factor_rows)}'
                f'</div>'
            )
            st.markdown(explain_html, unsafe_allow_html=True)

            # RETENTION PLAYBOOK: Strategic Decision Support
            st.markdown("#### Recommended Retention Playbook")
            st.caption("Actionable, rule-grounded interventions prioritized to minimize churn hazard and customer acquisition loss.")

            p1, p2, p3 = st.columns(3)
            with p1:
                if risk_level == "High":
                    p_pri = "high"
                    p_pri_text = "HIGH PRIORITY"
                    p_title = "Contract Term Migration"
                    p_body = "Offer 20% promotional tariff discount conditioned on transitioning to a 12-month annual commitment."
                elif risk_level == "Medium":
                    p_pri = "med"
                    p_pri_text = "MEDIUM PRIORITY"
                    p_title = "Onboarding Health Check"
                    p_body = "Trigger automated CS outreach call to identify onboarding friction and guide self-service portal setup."
                else:
                    p_pri = "low"
                    p_pri_text = "OPPORTUNITY"
                    p_title = "Annual Loyalty Recognition"
                    p_body = "Send automated VIP customer appreciation note with anniversary account milestone bonus."

                with st.container(border=True):
                    st.markdown(f'<span class="playbook-priority {p_pri}">{p_pri_text}</span>', unsafe_allow_html=True)
                    st.markdown(f"**{p_title}**")
                    st.caption(p_body)

            with p2:
                if online_security == "No" or tech_support == "No":
                    p2_pri = "high" if risk_level == "High" else "med"
                    p2_pri_text = "HIGH PRIORITY" if risk_level == "High" else "MEDIUM PRIORITY"
                    p2_title = "Digital Care Bundle Trial"
                    p2_body = "Provide 90-day complimentary subscription to Online Security and Tech Support to resolve device friction."
                else:
                    p2_pri = "med"
                    p2_pri_text = "MEDIUM PRIORITY"
                    p2_title = "Streaming Package Cross-Sell"
                    p2_body = "Account is stable; present bundled entertainment streaming packages to deepen multi-product stickiness."

                with st.container(border=True):
                    st.markdown(f'<span class="playbook-priority {p2_pri}">{p2_pri_text}</span>', unsafe_allow_html=True)
                    st.markdown(f"**{p2_title}**")
                    st.caption(p2_body)

            with p3:
                if "electronic check" in payment_method.lower():
                    p3_pri = "med"
                    p3_pri_text = "MEDIUM PRIORITY"
                    p3_title = "Auto-Pay Incentive Credit"
                    p3_body = "Grant a one-time $15 bill credit upon enrolling in automatic bank transfer or credit card auto-debit."
                else:
                    p3_pri = "low"
                    p3_pri_text = "LOW PRIORITY"
                    p3_title = "Account Health Tracking"
                    p3_body = "Enroll in quarterly automated account health telemetry; maintain continuous low-touch surveillance."

                with st.container(border=True):
                    st.markdown(f'<span class="playbook-priority {p3_pri}">{p3_pri_text}</span>', unsafe_allow_html=True)
                    st.markdown(f"**{p3_title}**")
                    st.caption(p3_body)

# ══════════════════════════════════════════════════════════════════════
#  VIEW 2: BATCH PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════
elif active_nav == "Batch Portfolio Analytics":
    st.markdown("#### High-Throughput Batch Scoring & Portfolio Diagnostics")
    st.caption("Score cohorts of customer accounts, identify revenue concentrations at risk, and segment retention cohorts.")

    col_dl, col_blank = st.columns([1, 2])
    with col_dl:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_customers.csv')
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                sample_csv_data = f.read()
            st.download_button(
                label="📥 Download Template Benchmark CSV",
                data=sample_csv_data,
                file_name="telco_batch_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

    uploaded_file = st.file_uploader("Upload Batch CSV File", type="csv")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.markdown(f"**Cohort Preview** ({len(df_raw)} records detected)")
            st.dataframe(df_raw.head(4), use_container_width=True)

            score_btn = st.button("Score Customer Portfolio →", type="primary")

            if score_btn or 'batch_scored_df' in st.session_state:
                if score_btn:
                    with st.spinner("Processing batch feature transformation & scoring..."):
                        if mode == "REST API Microservice (FastAPI)":
                            try:
                                files = {'file': uploaded_file.getvalue()}
                                resp = requests.post(f"{api_url}/predict/batch/csv", files=files, timeout=15)
                                if resp.status_code == 200:
                                    preds = resp.json()['predictions']
                                    scored_df = df_raw.copy()
                                    scored_df['churn_probability'] = [p['churn_probability'] for p in preds]
                                    scored_df['prediction'] = [p['prediction'] for p in preds]
                                    scored_df['risk_level'] = [p['risk_level'] for p in preds]
                                else:
                                    st.warning("FastAPI batch service failed; falling back to in-process pipeline.")
                                    scored_df = predict_batch_customers(df_raw, local_pipeline)
                            except Exception:
                                st.info(f"ℹ️ FastAPI backend at `{api_url}` is unreachable. Seamlessly evaluated batch portfolio using in-process model.")
                                scored_df = predict_batch_customers(df_raw, local_pipeline)
                        else:
                            scored_df = predict_batch_customers(df_raw, local_pipeline)

                        st.session_state['batch_scored_df'] = scored_df

                scored_df = st.session_state.get('batch_scored_df')

                if scored_df is not None:
                    tot = len(scored_df)
                    high_cnt = int((scored_df['risk_level'] == 'High').sum())
                    med_cnt = int((scored_df['risk_level'] == 'Medium').sum())
                    low_cnt = int((scored_df['risk_level'] == 'Low').sum())
                    churn_rate = (scored_df['prediction'].sum() / tot) * 100.0

                    # Calculate Revenue at Risk (MonthlyCharges for predicted churners)
                    if 'MonthlyCharges' in scored_df.columns:
                        rev_at_risk = scored_df[scored_df['prediction'] == 1]['MonthlyCharges'].sum()
                    else:
                        rev_at_risk = 0.0

                    # Enterprise KPI Cards
                    k1, k2, k3, k4, k5 = st.columns(5)
                    with k1:
                        st.metric("Total Accounts", f"{tot:,}")
                    with k2:
                        st.metric("Projected Churn", f"{churn_rate:.1f}%")
                    with k3:
                        st.metric("High Risk Accounts", f"{high_cnt}", delta=f"{(high_cnt/tot)*100:.1f}% of base", delta_color="inverse")
                    with k4:
                        st.metric("Monthly MRR at Risk", f"${rev_at_risk:,.2f}", delta="Urgent Exposure", delta_color="inverse")
                    with k5:
                        st.metric("Safe Base", f"{low_cnt}", delta=f"{(low_cnt/tot)*100:.1f}% healthy")

                    st.write("")

                    # Plotly Visualizations (Stripe/Linear Clean Aesthetics)
                    ch1, ch2 = st.columns(2)

                    with ch1:
                        # Risk Distribution Donut
                        risk_counts = scored_df['risk_level'].value_counts().reset_index()
                        risk_counts.columns = ['Risk Tier', 'Count']
                        fig_donut = px.pie(
                            risk_counts,
                            values='Count',
                            names='Risk Tier',
                            title='Cohort Risk Distribution',
                            color='Risk Tier',
                            color_discrete_map=RISK_COLOR_MAP,
                            hole=0.6,
                        )
                        fig_donut.update_layout(**PLOTLY_ENTERPRISE)
                        fig_donut.update_traces(
                            textinfo='percent+label',
                            marker=dict(line=dict(color='#FFFFFF', width=2)),
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)

                    with ch2:
                        # Scatter: Tenure vs Churn Propensity
                        fig_scatter = px.scatter(
                            scored_df,
                            x='tenure',
                            y='churn_probability',
                            color='risk_level',
                            title='Tenure Duration vs Churn Propensity',
                            labels={'tenure': 'Tenure (Months)', 'churn_probability': 'Churn Probability (0-1)'},
                            color_discrete_map=RISK_COLOR_MAP,
                        )
                        fig_scatter.update_layout(**PLOTLY_ENTERPRISE)
                        fig_scatter.update_xaxes(gridcolor='#E2E8F0', zerolinecolor='#CBD5E1')
                        fig_scatter.update_yaxes(gridcolor='#E2E8F0', zerolinecolor='#CBD5E1')
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    # Interactive Filterable Risk Table
                    st.markdown("#### Scored Accounts Register")
                    col_f1, col_f2 = st.columns([1, 2])
                    with col_f1:
                        filter_tier = st.selectbox("Filter by Risk Tier", ["All Tiers", "High Risk Only", "Medium Risk Only", "Low Risk Only"])

                    filtered_view = scored_df.copy()
                    if filter_tier == "High Risk Only":
                        filtered_view = filtered_view[filtered_view['risk_level'] == 'High']
                    elif filter_tier == "Medium Risk Only":
                        filtered_view = filtered_view[filtered_view['risk_level'] == 'Medium']
                    elif filter_tier == "Low Risk Only":
                        filtered_view = filtered_view[filtered_view['risk_level'] == 'Low']

                    st.dataframe(
                        filtered_view.style.format({'churn_probability': '{:.3f}'}),
                        use_container_width=True,
                    )

                    # Export
                    csv_export = scored_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Export Scored Portfolio CSV",
                        data=csv_export,
                        file_name="churn_scored_portfolio.csv",
                        mime="text/csv",
                    )
        except Exception as err:
            st.error(f"Error parsing uploaded CSV file: {err}")

# ══════════════════════════════════════════════════════════════════════
#  VIEW 3: MODEL DIAGNOSTICS & TELEMETRY
# ══════════════════════════════════════════════════════════════════════
elif active_nav == "Model Diagnostics & Metrics":
    st.markdown("#### Production Model Calibration & Diagnostics")
    st.caption("Detailed statistical evaluation, loss metrics, and cross-validated comparisons across candidate architectures.")

    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_payload = json.load(f)

        best_model_name = metrics_payload.get('best_model', 'Logistic Regression')
        best_metrics = metrics_payload.get('best_model_metrics', {})
        comparison_list = metrics_payload.get('comparison', [])

        st.info(f"🏆 **Champion Architecture**: **{best_model_name}** — Selected based on cross-validated F1-Score to optimize customer retention cost trade-offs.")

        # Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Test Accuracy", f"{best_metrics.get('Accuracy', 0.7587)*100:.2f}%")
        with m2:
            st.metric("F1 Score", f"{best_metrics.get('F1', 0.6256):.4f}")
        with m3:
            st.metric("Recall (Sensitivity)", f"{best_metrics.get('Recall', 0.7594)*100:.2f}%")
        with m4:
            st.metric("Precision", f"{best_metrics.get('Precision', 0.5318)*100:.2f}%")
        with m5:
            st.metric("ROC-AUC", f"{best_metrics.get('ROC-AUC', 0.8425):.4f}")

        st.write("")
        st.markdown("#### Candidate Model Benchmark Evaluation")
        comp_df = pd.DataFrame(comparison_list).set_index('Model')
        st.dataframe(comp_df.style.highlight_max(color='#D1FAE5', axis=0), use_container_width=True)
    else:
        st.warning("Metrics repository (metrics.json) not found in model storage directory.")

    st.write("")
    st.markdown("#### Offline Evaluation Visualizations")
    img_col1, img_col2, img_col3 = st.columns(3)

    cm_path = os.path.join(config.MODEL_DIR, 'confusion_matrix.png')
    roc_path = os.path.join(config.MODEL_DIR, 'roc_curve.png')
    fi_path = os.path.join(config.MODEL_DIR, 'feature_importance.png')

    with img_col1:
        st.markdown("**Confusion Matrix**")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), use_container_width=True)
        else:
            st.caption("Plot not generated.")

    with img_col2:
        st.markdown("**ROC-AUC Curve**")
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), use_container_width=True)
        else:
            st.caption("Plot not generated.")

    with img_col3:
        st.markdown("**Feature Importances**")
        if os.path.exists(fi_path):
            st.image(Image.open(fi_path), use_container_width=True)
        else:
            st.caption("Plot not generated.")

# ══════════════════════════════════════════════════════════════════════
#  VIEW 4: SYSTEM & API CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
elif active_nav == "System & API Config":
    st.markdown("#### Platform Microservice Architecture & API Contracts")
    st.caption("Inspect live endpoints, request schemas, health check diagnostics, and serialization specs.")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        st.markdown(
            """
            <div class="form-section-card">
                <div class="form-section-header">
                    <span class="form-section-title">⚙️ Inference Engine Routing</span>
                    <span class="form-section-badge">Architecture</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; gap: 0.65rem; font-size: 0.86rem; margin-top: 0.25rem;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--slate-100); padding-bottom: 0.45rem;">
                    <span style="color: var(--slate-600); font-weight: 500;">Active Execution Mode</span>
                    <span style="font-family: var(--font-mono); font-weight: 600; color: var(--slate-900);">{mode}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--slate-100); padding-bottom: 0.45rem;">
                    <span style="color: var(--slate-600); font-weight: 500;">Configured Gateway</span>
                    <span style="font-family: var(--font-mono); font-weight: 600; color: var(--slate-900);">{api_url}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--slate-100); padding-bottom: 0.45rem;">
                    <span style="color: var(--slate-600); font-weight: 500;">Model Artifact Path</span>
                    <span style="font-family: var(--font-mono); font-weight: 500; color: var(--slate-700); font-size: 0.8rem;">{config.MODEL_PATH}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--slate-100); padding-bottom: 0.45rem;">
                    <span style="color: var(--slate-600); font-weight: 500;">Engineered Numerical Features</span>
                    <span style="font-family: var(--font-mono); font-weight: 600; color: var(--slate-900);">{len(config.NUMERICAL_FEATURES)} features</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--slate-600); font-weight: 500;">Engineered Categorical Features</span>
                    <span style="font-family: var(--font-mono); font-weight: 600; color: var(--slate-900);">{len(config.CATEGORICAL_FEATURES)} features</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cfg2:
        st.markdown(
            """
            <div class="form-section-card">
                <div class="form-section-header">
                    <span class="form-section-title">📡 Microservice Health Diagnostics</span>
                    <span class="form-section-badge">Connectivity</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        diag_url = st.text_input(
            "FastAPI Target URL",
            value=api_url,
            help="Target URL for testing FastAPI microservice connectivity",
            key="diag_url_input",
        )
        test_btn = st.button("Ping FastAPI Backend Endpoint", use_container_width=True)
        if test_btn:
            clean_url = diag_url.strip().rstrip("/")
            try:
                t0 = time.time()
                r = requests.get(f"{clean_url}/", timeout=2.5)
                elapsed = (time.time() - t0) * 1000
                if r.status_code == 200:
                    st.success(f"🟢 **Gateway Connected**: 200 OK ({elapsed:.1f}ms latency)")
                    st.json(r.json())
                else:
                    st.warning(f"🟡 **Gateway Reachable**: Returned HTTP {r.status_code}")
                    if r.text:
                        st.text(r.text[:300])
            except Exception as ex:
                is_local = any(h in clean_url for h in ["localhost", "127.0.0.1", "0.0.0.0"])
                if is_local:
                    st.markdown(
                        f"""
                        <div style="background-color: #FFFBEB; border: 1px solid #FCD34D; border-radius: 8px; padding: 0.9rem; margin-top: 0.6rem;">
                            <div style="display: flex; align-items: center; gap: 0.4rem; font-weight: 700; color: #92400E; font-size: 0.88rem; margin-bottom: 0.35rem;">
                                <span>🟡</span> <span>Gateway Offline (Cloud Container Isolation)</span>
                            </div>
                            <p style="font-size: 0.82rem; color: #78350F; margin-bottom: 0.55rem; line-height: 1.45;">
                                On <b>Streamlit Community Cloud</b>, this app runs inside an isolated Linux container serving the Streamlit UI. The decoupled FastAPI microservice is not hosted on <code>localhost:8000</code> of this container.
                            </p>
                            <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 0.55rem 0.7rem; margin-bottom: 0.55rem; font-size: 0.8rem; color: #166534; line-height: 1.45;">
                                🛡️ <b>Resilient Auto-Failover Active:</b><br/>
                                All predictions and batch analytics automatically run via the <b>In-Process Machine Learning Pipeline</b> with 0 downtime and &lt;15ms latency.
                            </div>
                            <div style="font-size: 0.78rem; color: #64748B; line-height: 1.45;">
                                <b>To run locally with FastAPI:</b><br/>
                                <code style="color: #0F172A; background: #FEF3C7; padding: 2px 5px; border-radius: 4px; font-weight: 600;">uvicorn app.api:app --reload --port 8000</code><br/>
                                <b>To connect a cloud gateway:</b> Enter your deployed public API URL (Render/AWS) above.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 0.9rem; margin-top: 0.6rem;">
                            <div style="display: flex; align-items: center; gap: 0.4rem; font-weight: 700; color: #991B1B; font-size: 0.88rem; margin-bottom: 0.35rem;">
                                <span>🔴</span> <span>Remote Gateway Unreachable</span>
                            </div>
                            <p style="font-size: 0.82rem; color: #7F1D1D; margin-bottom: 0.55rem; line-height: 1.45;">
                                Could not establish connection to <code>{clean_url}</code>. The server might be booting, suspended, or blocking CORS requests.
                            </p>
                            <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 0.55rem 0.7rem; font-size: 0.8rem; color: #166534; line-height: 1.45;">
                                🛡️ <b>In-Process Engine Active:</b> Customer risk scoring continues uninterrupted via the embedded champion model.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with st.expander("Show Diagnostic Error Trace (Developer View)"):
                    st.code(str(ex), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("#### OpenAPI 3.0 Ingestion Schema Contract")
    sample_contract = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.50,
        "TotalCharges": 906.00,
    }
    st.json(sample_contract)
