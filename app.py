
import os
import zipfile
import tempfile

import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="MediScope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# Session State
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "theme" not in st.session_state:
    st.session_state.theme = "Light"


# =========================
# Theme Toggle
# =========================
top_left, top_right = st.columns([5, 1])
with top_right:
    st.session_state.theme = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if st.session_state.theme else "Light"

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.write("Use the switch at the top-right to change theme.")


# =========================
# CSS Themes
# =========================
if st.session_state.theme == "Dark":
    css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071f1b 0%, #0b2b25 45%, #101418 100%);
        color: #f5f7f7;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06251f 0%, #0b5c4a 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title h1, .section-title, .feature-card h3 {
        color: #38d9ad !important;
    }

    .main-title p, .small-muted, .feature-card p, .hero-card p {
        color: #d0d8d6 !important;
    }

    .hero-card, .feature-card, .section-card, .login-card {
        background: rgba(15, 35, 32, 0.96);
        border: 1px solid rgba(70, 220, 180, 0.18);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
        color: #f5f7f7;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #0e8d71, #28c99c);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #38d9ad;
    }
    
/* ===== DARK MODE VISIBILITY ===== */
div[data-testid="stTabs"] button, div[data-testid="stTabs"] button p {
    color: #d8fff5 !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
div[data-testid="stTabs"] [aria-selected="true"], div[data-testid="stTabs"] [aria-selected="true"] p {
    color: #38d9ad !important;
}
[data-testid="stMetricLabel"] {
    color: #d8fff5 !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
[data-testid="stMetricValue"] {
    color: #38d9ad !important;
}
label, .stRadio label, .stTextInput label, .stSlider label {
    color: #d8fff5 !important;
    font-weight: 700 !important;
}

    </style>
    """
else:
    css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef7f5 0%, #f8fbff 45%, #ffffff 100%);
        color: #1c2529;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b6b57 0%, #0f8f70 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title h1, .section-title, .feature-card h3 {
        color: #08785f !important;
    }

    .main-title p, .small-muted, .feature-card p, .hero-card p {
        color: #5b6670 !important;
    }

    .hero-card {
        background: linear-gradient(120deg, #d9f3dd, #f5fff7);
        border: 1px solid #c6e8cf;
        box-shadow: 0 18px 45px rgba(15, 143, 112, 0.15);
    }

    .feature-card, .section-card, .login-card {
        background: white;
        border: 1px solid #e8eeee;
        box-shadow: 0 12px 32px rgba(19, 48, 43, 0.08);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #08785f, #11a87f);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #08785f;
    }
    
/* ===== FIX VISIBILITY ===== */
div[data-testid="stTabs"] button {
    color: #08785f !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
div[data-testid="stTabs"] button p {
    color: #08785f !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    color: #ff4b4b !important;
    border-bottom: 3px solid #ff4b4b !important;
}
div[data-testid="stTabs"] [aria-selected="true"] p {
    color: #ff4b4b !important;
}
[data-testid="stMetricLabel"] {
    color: #08785f !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
[data-testid="stMetricValue"] {
    color: #08785f !important;
}
label, .stRadio label, .stTextInput label, .stSlider label {
    color: #08785f !important;
    font-weight: 700 !important;
}

    </style>
    """

st.markdown(css, unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    text-align: center;
    padding: 28px 16px 10px 16px;
}

.main-title h1 {
    font-size: 58px;
    font-weight: 800;
    margin-bottom: 0;
}

.main-title p {
    font-size: 18px;
    margin-top: 6px;
}

.hero-card {
    border-radius: 28px;
    padding: 34px;
    text-align: center;
    margin-bottom: 22px;
}

.hero-card h2 {
    color: #08785f;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}

.feature-card {
    border-radius: 22px;
    padding: 24px;
    min-height: 190px;
    transition: all 0.2s ease;
}

.feature-card h3 {
    font-weight: 800;
    font-size: 22px;
}

.feature-card p {
    font-size: 15px;
    line-height: 1.6;
}

.section-card, .login-card {
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 22px;
}

.section-title {
    font-weight: 800;
    font-size: 28px;
    margin-bottom: 8px;
}

.small-muted {
    font-size: 15px;
}

.login-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #08785f;
}

.login-subtitle {
    text-align: center;
    color: #6a777d;
    margin-bottom: 25px;
}

div.stButton > button:hover {
    filter: brightness(0.95);
    color: white;
}

hr {
    border: none;
    height: 1px;
    background: #d7ebe5;
    margin: 20px 0;
}

/* ===== PROFESSIONAL CENTERED TABS ===== */
div[data-testid="stTabs"] {
    width: 100%;
}

div[data-testid="stTabs"] div[role="tablist"] {
    justify-content: center !important;
    gap: 18px !important;
    border-bottom: 1px solid rgba(8, 120, 95, 0.18);
    padding-bottom: 12px;
}

div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 999px !important;
    padding: 10px 22px !important;
    min-width: 150px;
    background: rgba(255, 255, 255, 0.10) !important;
    border: 1px solid rgba(8, 120, 95, 0.18) !important;
}

div[data-testid="stTabs"] button[role="tab"] p {
    font-size: 16px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #08785f, #12b98d) !important;
    border: 1px solid transparent !important;
    box-shadow: 0 10px 24px rgba(8, 120, 95, 0.25);
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
    color: white !important;
}

/* ===== DASHBOARD METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(8, 120, 95, 0.13);
    border-radius: 22px;
    padding: 22px 24px;
    box-shadow: 0 12px 28px rgba(19, 48, 43, 0.08);
}

[data-testid="stMetricLabel"] {
    font-size: 16px !important;
}

[data-testid="stMetricValue"] {
    font-size: 34px !important;
    font-weight: 900 !important;
}


/* ===== MEDISCOPE LOGO ===== */
.brand-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 18px;
    padding: 24px 16px 8px 16px;
}

.logo-mark {
    width: 82px;
    height: 82px;
    border-radius: 26px;
    background: linear-gradient(135deg, #08785f, #13c99a);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 18px 38px rgba(8, 120, 95, 0.28);
    position: relative;
}

.logo-mark::before {
    content: "✚";
    color: white;
    font-size: 34px;
    font-weight: 900;
    position: absolute;
    top: 14px;
    left: 18px;
}

.logo-mark::after {
    content: "⌕";
    color: rgba(255,255,255,0.92);
    font-size: 42px;
    font-weight: 900;
    position: absolute;
    right: 12px;
    bottom: 5px;
    transform: rotate(-12deg);
}

.brand-text h1 {
    font-size: 58px;
    font-weight: 900;
    margin: 0;
    color: #08785f;
    letter-spacing: -1px;
}

.brand-text p {
    font-size: 18px;
    color: #5b6670;
    margin: 4px 0 0 2px;
}

.login-logo {
    display: flex;
    justify-content: center;
    margin-bottom: 12px;
}

.sidebar-brand {
    text-align: center;
    padding: 10px 0 16px 0;
}

.sidebar-logo {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    background: linear-gradient(135deg, #ffffff, #d8fff5);
    color: #08785f;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 900;
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    margin-bottom: 8px;
}

.sidebar-brand-title {
    font-size: 22px;
    font-weight: 900;
}


/* ===== ANIMATED MEDISCOPE LOGO - FINAL ===== */
.brand-wrap {
    display: block !important;
    text-align: center !important;
    padding: 28px 16px 12px 16px !important;
}

.logo-mark {
    width: 112px !important;
    height: 112px !important;
    margin: 0 auto 14px auto !important;
    border-radius: 32px !important;
    background: linear-gradient(135deg, #08785f, #00d4a4, #38d9ad) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 20px 45px rgba(8, 120, 95, 0.35) !important;
    position: relative !important;
    animation: floatLogo 3.2s ease-in-out infinite, glowPulse 2.4s ease-in-out infinite !important;
    overflow: visible !important;
}

.logo-mark::before {
    content: "🔬" !important;
    position: static !important;
    color: white !important;
    font-size: 48px !important;
    font-weight: 900 !important;
    filter: drop-shadow(0 6px 10px rgba(0,0,0,0.25)) !important;
    animation: iconPulse 2.2s ease-in-out infinite !important;
}

.logo-mark::after {
    content: "" !important;
    position: absolute !important;
    width: 150px !important;
    height: 150px !important;
    border-radius: 42px !important;
    background: radial-gradient(circle, rgba(0, 212, 164, 0.35), transparent 68%) !important;
    z-index: -1 !important;
    animation: auraPulse 2.4s ease-in-out infinite !important;
}

.brand-text h1 {
    font-size: 64px !important;
    font-weight: 900 !important;
    margin: 0 !important;
    letter-spacing: -1.5px !important;
    background: linear-gradient(90deg, #08785f, #00c9a7, #38d9ad) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.brand-text p {
    font-size: 18px !important;
    color: #5b6670 !important;
    margin-top: 6px !important;
}

.sidebar-logo {
    animation: smallPulse 2.5s ease-in-out infinite !important;
}

@keyframes floatLogo {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-8px) rotate(-1deg); }
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 20px 45px rgba(8, 120, 95, 0.30); }
    50% { box-shadow: 0 25px 60px rgba(0, 212, 164, 0.55); }
}

@keyframes auraPulse {
    0%, 100% { transform: scale(0.92); opacity: 0.55; }
    50% { transform: scale(1.18); opacity: 0.95; }
}

@keyframes iconPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.12); }
}

@keyframes smallPulse {
    0%, 100% { transform: scale(1); box-shadow: 0 10px 24px rgba(0,0,0,0.16); }
    50% { transform: scale(1.06); box-shadow: 0 14px 34px rgba(0,0,0,0.24); }
}


/* ===== CREATIVE MEDISCOPE LOGO - SHIELD + DOC + HEARTBEAT + SEARCH ===== */
.logo-mark {
    width: 122px !important;
    height: 122px !important;
    margin: 0 auto 16px auto !important;
    border-radius: 34px !important;
    background:
        radial-gradient(circle at 30% 25%, rgba(255,255,255,0.34), transparent 28%),
        linear-gradient(135deg, #064f46 0%, #08a87f 45%, #22e6b6 100%) !important;
    position: relative !important;
    box-shadow:
        0 24px 55px rgba(8, 120, 95, 0.36),
        inset 0 1px 0 rgba(255,255,255,0.38) !important;
    animation: floatLogo 3.2s ease-in-out infinite, glowPulse 2.4s ease-in-out infinite !important;
    overflow: visible !important;
}

/* remove old microscope emoji */
.logo-mark::before {
    content: "" !important;
    position: absolute !important;
    width: 58px !important;
    height: 72px !important;
    left: 29px !important;
    top: 22px !important;
    background: rgba(255,255,255,0.96) !important;
    border-radius: 16px 16px 20px 20px !important;
    clip-path: polygon(50% 0%, 100% 18%, 92% 72%, 50% 100%, 8% 72%, 0% 18%) !important;
    box-shadow: 0 12px 22px rgba(0,0,0,0.16) !important;
    animation: iconPulse 2.2s ease-in-out infinite !important;
}

/* search lens ring */
.logo-mark::after {
    content: "" !important;
    position: absolute !important;
    width: 42px !important;
    height: 42px !important;
    right: 20px !important;
    bottom: 22px !important;
    border: 8px solid rgba(255,255,255,0.92) !important;
    border-radius: 50% !important;
    background: transparent !important;
    box-shadow: 0 10px 18px rgba(0,0,0,0.18) !important;
    z-index: 3 !important;
    animation: lensMove 3.2s ease-in-out infinite !important;
}

/* heartbeat line inside logo */
.logo-mark .pulse-line {
    display: none;
}

.brand-wrap::after {
    content: "⌁";
    position: absolute;
    left: calc(50% - 42px);
    top: 68px;
    font-size: 56px;
    color: #08785f;
    z-index: 8;
    font-weight: 900;
    transform: rotate(0deg);
    opacity: 0.95;
    animation: heartbeat 1.8s ease-in-out infinite;
}

/* magnifier handle */
.brand-wrap::before {
    content: "";
    position: absolute;
    left: calc(50% + 46px);
    top: 116px;
    width: 34px;
    height: 9px;
    border-radius: 999px;
    background: rgba(255,255,255,0.92);
    transform: rotate(45deg);
    z-index: 8;
    box-shadow: 0 8px 14px rgba(0,0,0,0.16);
}

.brand-wrap {
    position: relative !important;
}

@keyframes lensMove {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(-4px, -3px) scale(1.04); }
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); opacity: 0.92; }
    35% { transform: scale(1.12); opacity: 1; }
    55% { transform: scale(0.98); opacity: 0.95; }
}

</style>
""", unsafe_allow_html=True)


# =========================
# Login Page
# =========================
def login_page():
    st.markdown("""
    <div class="brand-wrap">
        <div class="logo-mark"></div>
        <div class="brand-text">
            <h1>MediScope</h1>
            <p>Smart Hospital Document Intelligence System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown("""
        <div class="login-card">
            <div class="login-title">Login</div>
            <div class="login-subtitle">Enter your username and password to access the system</div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_clicked = st.button("Login", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if login_clicked:
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Wrong username or password")

        st.info("Default login: username = admin | password = 1234")


if not st.session_state.logged_in:
    login_page()
    st.stop()


# =========================
# Header
# =========================
st.markdown("""
<div class="brand-wrap">
    <div class="logo-mark"></div>
    <div class="brand-text">
        <h1>MediScope</h1>
        <p>Smart Hospital Document Intelligence System</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">✚</div>
        <div class="sidebar-brand-title">MediScope</div>
        <div>Document Intelligence GUI</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()


# =========================
# NLTK setup
# =========================
@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

setup_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# =========================
# Functions
# =========================
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def extract_uploaded_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "Data.zip")

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, dirs, files in os.walk(temp_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        if txt_files:
            return root

    return None


def load_documents(data_path):
    data = []
    names = []

    for file in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, file)

        if os.path.isfile(path) and file.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data.append(f.read())
                names.append(file)

    return data, names


# =========================
# Data
# =========================
with st.sidebar:
    uploaded_file = st.file_uploader("Optional: Upload Data.zip", type="zip")

if uploaded_file is not None:
    data_path = extract_uploaded_zip(uploaded_file)
else:
    data_path = "Data"

if not os.path.exists(data_path):
    st.error("Data folder not found. Put a folder named Data beside app.py, or upload Data.zip from the sidebar.")
    st.stop()

Data, filenames = load_documents(data_path)

if len(Data) == 0:
    st.error("No .txt files found inside Data folder.")
    st.stop()

cleaned_docs = [preprocess(doc) for doc in Data]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_docs)

st.sidebar.success(f"Loaded {len(Data)} documents")


# =========================
# Tabs
# =========================
home_tab, retrieval_tab, classification_tab, clustering_tab = st.tabs(
    ["🏠 Home", "🔍 Information Retriever", "🧠 Classification", "📊 Clustering"]
)


# =========================
# Home
# =========================
with home_tab:
    st.markdown("""
    <div class="hero-card">
        <h2>🔬 Welcome to MediScope</h2>
        <p>
            A professional hospital document intelligence system for semantic retrieval,
            recommendation classification, and document clustering using NLP and machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("📄 Total Documents", len(Data))
    m2.metric("⚙️ System Functions", "3 Tools")
    m3.metric("🧠 AI Technique", "NLP")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Retrieve</h3>
            <p>Enter a query and retrieve the most similar hospital documents using TF-IDF and cosine similarity.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Classify</h3>
            <p>Classify documents into recommendation labels using Naive Bayes and Decision Tree models.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Cluster</h3>
            <p>Group documents automatically into clusters using K-Means clustering.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Loaded Documents</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"File Name": filenames}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Retrieval
# =========================
with retrieval_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Information Retriever</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Search hospital documents based on semantic similarity.</p>', unsafe_allow_html=True)

    query = st.text_input("Search Query", value="clean hospital and friendly nurses")
    top_k = st.slider("Top documents", 1, len(filenames), min(10, len(filenames)))

    if st.button("🔍 Search Documents", key="search_btn"):
        cleaned_query = preprocess(query)
        query_vector = tfidf_vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        results = pd.DataFrame({
            "File Name": filenames,
            "Similarity Score": similarities
        }).sort_values(by="Similarity Score", ascending=False)

        st.write("### Result List")
        st.dataframe(results.head(top_k), use_container_width=True)

        relevant_docs = {name for name in filenames if name.startswith("hospital")}
        retrieved_docs = results.head(top_k)["File Name"].tolist()

        tp = len(set(retrieved_docs) & relevant_docs)
        precision = tp / len(retrieved_docs) if len(retrieved_docs) else 0
        recall = tp / len(relevant_docs) if len(relevant_docs) else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("True Positives", tp)
        m2.metric("Precision", round(precision, 2))
        m3.metric("Recall", round(recall, 2))

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Classification
# =========================
with classification_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Classification</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Predict recommendation labels for documents.</p>', unsafe_allow_html=True)

    label_map = {
        "hospital1": "Conditional_rec",
        "hospital2": "Better_rec",
        "hospital3": "Conditional_rec",
        "hospital4": "Least_rec",
        "hospital5": "Better_rec",
        "hospital6": "Conditional_rec",
        "hospital7": "Least_rec",
        "hospital8": "Better_rec",
        "hospital9": "Conditional_rec",
        "hospital10": "Better_rec",
        "hospital11": "Better_rec",
        "hospital12": "Conditional_rec",
        "hospital13": "Conditional_rec",
        "hospital14": "Least_rec",
        "hospital15": "Least_rec"
    }

    if st.button("🧠 Run Classification", key="classify_btn"):
        train_docs, train_names = [], []
        test_docs, test_names = [], []

        for file in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, file)

            if os.path.isfile(path) and file.endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                name = file.replace(".txt", "")

                if file.startswith("hospital"):
                    train_docs.append(text)
                    train_names.append(name)
                else:
                    test_docs.append(text)
                    test_names.append(name)

        clean_train = [preprocess(d) for d in train_docs]
        clean_test = [preprocess(d) for d in test_docs]
        y_train = [label_map[n] for n in train_names if n in label_map]

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(clean_train)
        X_test = vectorizer.transform(clean_test)

        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)

        st.write("### Naive Bayes Output")
        st.dataframe(pd.DataFrame({
            "Document": test_names,
            "Predicted Label": y_pred_nb
        }), use_container_width=True)

        nb_train_pred = nb.predict(X_train)
        report = classification_report(y_train, nb_train_pred, output_dict=True)

        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        cm = confusion_matrix(y_train, nb_train_pred, labels=nb.classes_)
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=nb.classes_)
        disp.plot(ax=ax)
        ax.set_title("Naive Bayes Confusion Matrix")
        st.pyplot(fig)

        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        dt_train_pred = dt.predict(X_train)

        st.write("### Decision Tree Output")
        st.dataframe(pd.DataFrame({
            "Document": test_names,
            "Predicted Label": y_pred_dt
        }), use_container_width=True)

        st.metric("Decision Tree Train Accuracy", round(accuracy_score(y_train, dt_train_pred), 2))

        fig2, ax2 = plt.subplots(figsize=(18, 8))
        plot_tree(
            dt,
            feature_names=vectorizer.get_feature_names_out(),
            class_names=dt.classes_,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=8,
            ax=ax2
        )
        ax2.set_title("Decision Tree")
        st.pyplot(fig2)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Clustering
# =========================
with clustering_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Clustering</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Group documents automatically using K-Means.</p>', unsafe_allow_html=True)

    if st.button("📊 Run Clustering", key="cluster_btn"):
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)

        clusters = kmeans.labels_

        results = pd.DataFrame({
            "File Name": filenames,
            "Cluster": clusters
        })

        st.write("### Clustering Results")
        st.dataframe(results, use_container_width=True)

        true_labels = []
        for doc in cleaned_docs:
            if "good" in doc or "great" in doc or "excellent" in doc or "amazing" in doc:
                true_labels.append(1)
            else:
                true_labels.append(0)

        acc1 = accuracy_score(true_labels, clusters)
        reversed_clusters = [1 - c for c in clusters]
        acc2 = accuracy_score(true_labels, reversed_clusters)
        final_accuracy = max(acc1, acc2)

        st.metric("Clustering Accuracy", round(final_accuracy, 2))

        cluster_count = results["Cluster"].value_counts().reset_index()
        cluster_count.columns = ["Cluster", "Count"]

        st.write("### Cluster Distribution")
        st.bar_chart(cluster_count.set_index("Cluster"))

    st.markdown("</div>", unsafe_allow_html=True)
