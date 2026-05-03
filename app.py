import streamlit as st
import subprocess

st.set_page_config(page_title="Document System", layout="wide")

st.title("🏥 Document Management System")

st.sidebar.title("Menu")
choice = st.sidebar.selectbox(
    "Choose Function",
    ["Home", "Run Project"]
)

if choice == "Home":
    st.write("Welcome 👋")
    st.write("Click 'Run Project' to execute your system")

elif choice == "Run Project":

    st.subheader("Run Your Code")

    if st.button("▶️ Run"):

        with st.spinner("Running..."):

            result = subprocess.run(
                ["python", "Pasted code(1).py"],
                capture_output=True,
                text=True
            )

        st.success("Finished ✅")

        st.subheader("Output:")
        st.text(result.stdout)

        if result.stderr:
            st.error(result.stderr)
