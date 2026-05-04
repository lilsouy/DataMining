
import os
import zipfile
import tempfile
import base64

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
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# SVG Logo
# =========================
LOGO_SVG = r"""
<svg class="mediscope-svg-logo" viewBox="0 0 620 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MediScope logo">
  <defs>
    <linearGradient id="shieldGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#79d894"/>
      <stop offset="48%" stop-color="#18b894"/>
      <stop offset="100%" stop-color="#0a6f63"/>
    </linearGradient>

    <linearGradient id="shieldDeep" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f9677"/>
      <stop offset="100%" stop-color="#07564e"/>
    </linearGradient>

    <filter id="softGlow" x="-35%" y="-35%" width="170%" height="170%">
      <feGaussianBlur stdDeviation="3.5" result="blur"/>
      <feColorMatrix in="blur" type="matrix"
        values="0 0 0 0 0.00
                0 0 0 0 0.85
                0 0 0 0 0.65
                0 0 0 0.55 0"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <filter id="textShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="1.2" flood-color="#042f2a" flood-opacity="0.55"/>
    </filter>
  </defs>

  <!-- icon group -->
  <g transform="translate(15,18)" filter="url(#softGlow)">
    <!-- outer shield -->
    <path d="M82 5
             C104 19 126 24 151 28
             C156 29 160 34 160 40
             L160 86
             C160 124 131 151 82 169
             C33 151 4 124 4 86
             L4 40
             C4 34 8 29 13 28
             C38 24 60 19 82 5Z"
          fill="url(#shieldGrad)" stroke="#eafff8" stroke-width="3"/>

    <!-- lower dark sweep -->
    <path d="M16 118
             C52 142 105 138 148 101
             C139 132 111 154 82 166
             C52 154 29 138 16 118Z"
          fill="url(#shieldDeep)" opacity="0.92"/>

    <!-- left chart bars -->
    <g fill="#f7fffb" opacity="0.96">
      <rect x="29" y="80" width="10" height="38" rx="3"/>
      <rect x="45" y="66" width="10" height="52" rx="3"/>
      <rect x="61" y="92" width="10" height="26" rx="3"/>
    </g>

    <!-- medical cross -->
    <g fill="#f7fffb" opacity="0.98">
      <rect x="70" y="38" width="16" height="48" rx="7"/>
      <rect x="54" y="54" width="48" height="16" rx="7"/>
    </g>

    <!-- magnifier circle -->
    <circle cx="90" cy="88" r="38" fill="#eafff8" opacity="0.94"/>
    <circle cx="90" cy="88" r="27" fill="none" stroke="#089277" stroke-width="9"/>
    <circle cx="90" cy="88" r="35" fill="none" stroke="#ffffff" stroke-width="5"/>

    <!-- magnifier handle -->
    <line x1="118" y1="116" x2="150" y2="148"
          stroke="#eafff8" stroke-width="14" stroke-linecap="round"/>
    <line x1="118" y1="116" x2="150" y2="148"
          stroke="#067463" stroke-width="7" stroke-linecap="round"/>

    <!-- white outline accent -->
    <path d="M24 42 C45 38 64 32 82 21 C101 32 120 38 141 42"
          fill="none" stroke="#ffffff" stroke-width="4" stroke-linecap="round" opacity="0.9"/>
  </g>

  <!-- text -->
  <g transform="translate(205,55)" filter="url(#textShadow)">
    <text x="0" y="55"
          font-family="Inter, Segoe UI, Arial, sans-serif"
          font-size="58"
          font-weight="800"
          letter-spacing="-1"
          fill="#eafff8"
          stroke="#07564e"
          stroke-width="2.4"
          paint-order="stroke fill">
      MediScope
    </text>

    <text x="4" y="95"
          font-family="Inter, Segoe UI, Arial, sans-serif"
          font-size="24"
          font-weight="700"
          letter-spacing="9"
          fill="#d9e7e3"
          opacity="0.95">
      MEDICAL HEALTH
    </text>
  </g>
</svg>
"""


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
    dark_on = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if dark_on else "Light"


# =========================
# CSS
# =========================
if st.session_state.theme == "Dark":
    app_background = "linear-gradient(135deg, #071f1b 0%, #0b2b25 45%, #101418 100%)"
    card_background = "rgba(15, 35, 32, 0.96)"
    text_main = "#f5f7f7"
    text_muted = "#d0d8d6"
    accent = "#38d9ad"
    tab_bg = "rgba(255,255,255,0.08)"
    metric_bg = "rgba(15, 35, 32, 0.90)"
    bg_opacity = "0.055"
else:
    app_background = "linear-gradient(135deg, #eef7f5 0%, #f8fbff 45%, #ffffff 100%)"
    card_background = "rgba(255, 255, 255, 0.92)"
    text_main = "#1c2529"
    text_muted = "#5b6670"
    accent = "#08785f"
    tab_bg = "rgba(255,255,255,0.76)"
    metric_bg = "rgba(255,255,255,0.76)"
    bg_opacity = "0.055"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {app_background};
    color: {text_main};
}}

/* Logo as soft background watermark */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image: none;
    background-repeat: no-repeat;
    background-position: center 330px;
    background-size: 760px;
    opacity: {bg_opacity};
    filter: blur(0.2px);
}}

.block-container {{
    position: relative;
    z-index: 1;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #064f46 0%, #0b8a6b 100%);
}}

[data-testid="stSidebar"] * {{
    color: white !important;
}}

.logo-header {{
    text-align: center;
    padding: 28px 0 18px 0;
}}

.logo-header img {{ display:none; }}

.logo-watermark {{
    position: fixed;
    left: 50%;
    top: 54%;
    transform: translate(-50%, -50%);
    width: 820px;
    max-width: 90vw;
    opacity: 0.045;
    z-index: 0;
    pointer-events: none;
}}
.logo-watermark svg {{
    width: 100%;
    height: auto;
}}
.logo-header svg {{
    width: 420px;
    max-width: 88%;
    height: auto;
    filter: drop-shadow(0 18px 35px rgba(8,120,95,0.28));
}}
.sidebar-logo-img svg {{
    width: 165px;
    max-width: 88%;
    height: auto;
    filter: drop-shadow(0 8px 16px rgba(0,0,0,0.18));
}}

.logo-header img {{
    width: 390px;
    max-width: 85%;
    border-radius: 18px;
    filter: drop-shadow(0 18px 35px rgba(8,120,95,0.20));
}}

.sidebar-logo-img {{
    width: 150px;
    max-width: 85%;
    border-radius: 14px;
    margin: 6px auto 10px auto;
    display: block;
}}

.login-card, .section-card, .feature-card, .hero-card {{
    background: {card_background};
    border: 1px solid rgba(8, 120, 95, 0.15);
    box-shadow: 0 18px 42px rgba(19, 48, 43, 0.09);
    border-radius: 28px;
}}

.login-card {{
    padding: 34px;
    margin-top: 10px;
}}

.login-title {{
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    color: {accent};
}}

.login-subtitle {{
    text-align: center;
    color: {text_muted};
    margin-bottom: 24px;
}}

.hero-card {{
    padding: 38px;
    text-align: center;
    margin-bottom: 26px;
}}

.hero-card h2 {{
    color: {accent};
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 8px;
}}

.hero-card p {{
    color: {text_muted};
    font-size: 17px;
    line-height: 1.7;
}}

.feature-card {{
    padding: 26px;
    min-height: 180px;
    transition: 0.2s ease;
}}

.feature-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 22px 50px rgba(8, 120, 95, 0.16);
}}

.feature-card h3 {{
    color: {accent};
    font-size: 23px;
    font-weight: 900;
}}

.feature-card p {{
    color: {text_muted};
    line-height: 1.6;
}}

.section-card {{
    padding: 28px;
    margin-bottom: 24px;
}}

.section-title {{
    color: {accent};
    font-size: 30px;
    font-weight: 900;
    margin-bottom: 8px;
}}

.small-muted {{
    color: {text_muted};
    font-size: 15px;
}}

div.stButton > button {{
    background: linear-gradient(90deg, #08785f, #10b98c);
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.72rem 1.55rem;
    font-weight: 800;
    box-shadow: 0 12px 24px rgba(8, 120, 95, 0.22);
}}

div.stButton > button:hover {{
    background: linear-gradient(90deg, #076b55, #0e906d);
    color: white;
}}

label, .stTextInput label, .stSlider label {{
    color: {accent} !important;
    font-weight: 800 !important;
}}

/* centered tabs */
div[data-testid="stTabs"] div[role="tablist"] {{
    justify-content: center !important;
    gap: 18px !important;
    border-bottom: 1px solid rgba(8, 120, 95, 0.18);
    padding-bottom: 12px;
}}

div[data-testid="stTabs"] button[role="tab"] {{
    border-radius: 999px !important;
    padding: 10px 22px !important;
    min-width: 155px;
    background: {tab_bg} !important;
    border: 1px solid rgba(8, 120, 95, 0.18) !important;
}}

div[data-testid="stTabs"] button[role="tab"] p {{
    color: {accent} !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    margin: 0 !important;
}}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: linear-gradient(90deg, #08785f, #12b98d) !important;
    border: 1px solid transparent !important;
    box-shadow: 0 10px 24px rgba(8, 120, 95, 0.25);
}}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{
    color: white !important;
}}

[data-testid="stMetric"] {{
    background: {metric_bg};
    border: 1px solid rgba(8, 120, 95, 0.13);
    border-radius: 22px;
    padding: 22px 24px;
    box-shadow: 0 12px 28px rgba(19, 48, 43, 0.08);
}}

[data-testid="stMetricLabel"] {{
    color: {accent} !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}}

[data-testid="stMetricValue"] {{
    color: {accent} !important;
    font-size: 34px !important;
    font-weight: 900 !important;
}}

.sidebar-brand {{
    text-align: center;
    padding: 10px 0 16px 0;
}}

.sidebar-brand-title {{
    font-size: 22px;
    font-weight: 900;
}}

hr {{
    border: none;
    height: 1px;
    background: rgba(8, 120, 95, 0.18);
    margin: 20px 0;
}}
</style>
""", unsafe_allow_html=True)



st.markdown(f"""
<div class="logo-watermark">
    {LOGO_SVG}
</div>
""", unsafe_allow_html=True)

# =========================
# Reusable Header
# =========================
def render_header():
    st.markdown(f"""
    <div class="logo-header">
        {LOGO_SVG}
    </div>
    """, unsafe_allow_html=True)


# =========================
# Login Page
# =========================
def login_page():
    render_header()

    left, center, right = st.columns([1, 1.15, 1])

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
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.write("Use the switch at the top-right to change theme.")
    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="sidebar-logo-img">{LOGO_SVG}</div>
        <div class="sidebar-brand-title">MediScope</div>
        <div>Document Intelligence GUI</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()


# =========================
# Main Header
# =========================
render_header()


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
        <h2>Welcome to MediScope</h2>
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
