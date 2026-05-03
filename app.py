
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
# Page settings
# =========================
st.set_page_config(
    page_title="Hospital Document Management System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Document Management System")
st.write("Retrieve, classify, and cluster documents using the same project logic.")


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
# Original preprocessing idea
# =========================
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# =========================
# Load data
# =========================
def extract_uploaded_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()

    zip_path = os.path.join(temp_dir, "Data.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find folder that contains txt files
    for root, dirs, files in os.walk(temp_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        if txt_files:
            return root

    return None


def load_documents(data_path):
    Data = []
    filenames = []

    for file in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, file)

        if os.path.isfile(path) and file.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                Data.append(f.read())
                filenames.append(file)

    return Data, filenames


# =========================
# Sidebar upload
# =========================
st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload Data.zip", type="zip")

if uploaded_file is not None:
    data_path = extract_uploaded_zip(uploaded_file)
else:
    data_path = "Data"

if not os.path.exists(data_path):
    st.warning("Please upload Data.zip or put a Data folder beside app.py.")
    st.stop()

Data, filenames = load_documents(data_path)

if len(Data) == 0:
    st.error("No .txt files found in the Data folder.")
    st.stop()

cleaned_docs = [preprocess(doc) for doc in Data]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_docs)

st.sidebar.success(f"Total files: {len(Data)}")


# =========================
# Tabs like the old GUI idea
# =========================
home_tab, retrieval_tab, classification_tab, clustering_tab = st.tabs(
    ["🏠 Home", "🔍 Retrieve Similar Documents", "🧠 Classify Documents", "📊 Group Documents"]
)


# =========================
# Home
# =========================
with home_tab:
    st.subheader("Welcome")
    st.write(
        """
        This GUI follows the same idea as the previous project interface:
        - Search for similar documents
        - Classify documents
        - Group documents into clusters
        """
    )

    st.info("Use the tabs above to run each project part.")

    st.write("### Loaded Files")
    st.dataframe(pd.DataFrame({"File Name": filenames}), use_container_width=True)


# =========================
# Retrieval
# =========================
with retrieval_tab:
    st.subheader("Retrieve Similar Documents")

    query = st.text_input(
        "Enter search query",
        value="clean hospital and friendly nurses"
    )

    top_k = st.slider("Number of top documents", 1, len(filenames), min(10, len(filenames)))

    if st.button("Search", key="search_btn"):
        cleaned_query = preprocess(query)
        query_vector = tfidf_vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        results = pd.DataFrame({
            "File Name": filenames,
            "Similarity Score": similarities
        })

        results = results.sort_values(by="Similarity Score", ascending=False)

        st.write("### Search Results")
        st.dataframe(results.head(top_k), use_container_width=True)

        relevant_docs = {name for name in filenames if name.startswith("hospital")}
        retrieved_docs = results.head(top_k)["File Name"].tolist()

        tp = len(set(retrieved_docs) & relevant_docs)
        precision = tp / len(retrieved_docs) if len(retrieved_docs) else 0
        recall = tp / len(relevant_docs) if len(relevant_docs) else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("True Positives", tp)
        c2.metric("Precision", round(precision, 2))
        c3.metric("Recall", round(recall, 2))


# =========================
# Classification
# =========================
with classification_tab:
    st.subheader("Classify Documents")

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

    if st.button("Run Classification", key="classify_btn"):
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

        # Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)

        results = pd.DataFrame({
            "Document": test_names,
            "Predicted Label - Naive Bayes": y_pred_nb
        })

        st.write("### Naive Bayes Predictions")
        st.dataframe(results, use_container_width=True)

        nb_train_pred = nb.predict(X_train)

        st.write("### Naive Bayes Train Report")
        report = classification_report(y_train, nb_train_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        cm = confusion_matrix(y_train, nb_train_pred, labels=nb.classes_)
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=nb.classes_)
        disp.plot(ax=ax)
        ax.set_title("Naive Bayes Confusion Matrix (Train)")
        st.pyplot(fig)

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)

        dt_results = pd.DataFrame({
            "Document": test_names,
            "Predicted Label - Decision Tree": y_pred_dt
        })

        st.write("### Decision Tree Predictions")
        st.dataframe(dt_results, use_container_width=True)

        dt_train_pred = dt.predict(X_train)
        st.metric("Decision Tree Accuracy (Train)", round(accuracy_score(y_train, dt_train_pred), 2))

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


# =========================
# Clustering
# =========================
with clustering_tab:
    st.subheader("Group Documents / Clustering")

    if st.button("Run Clustering", key="cluster_btn"):
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

        st.write("### Cluster Count")
        cluster_count = results["Cluster"].value_counts().reset_index()
        cluster_count.columns = ["Cluster", "Count"]
        st.bar_chart(cluster_count.set_index("Cluster"))
