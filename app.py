import streamlit as st
from backend.rag_pipeline import RAGPipeline
from chatbot_ollama import chatbot_answer

# ---------- Page Config ----------
st.set_page_config(
    page_title="PubMed Semantic Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS Styling ----------
st.markdown("""
        <style>
        /* Background & text */
        body {
            background-color: #f0f2f6;
            color: #1f2937;
            font-family: "Segoe UI", Roboto, Arial, sans-serif;
        }

        /* Title */
        .title-style {
            font-size: 42px;
            font-weight: 800;
            color: #1e3a8a;
            text-align: center;
            padding: 20px;
        }

        /* Input box */
        .stTextInput input {
            border-radius: 14px;
            border: 2px solid #3b82f6;
            padding: 12px;
            font-size: 16px;
            transition: all 0.3s ease-in-out;
        }
        .stTextInput input:focus {
            border-color: #1e40af;
            box-shadow: 0 0 8px rgba(30,64,175,0.3);
        }

        /* Buttons */
        .stButton>button {
            border-radius: 12px;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: 600;
            background-color: #2563eb;
            color: white;
            border: none;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #1e40af;
            transform: translateY(-2px);
        }

        /* Article card */
        .card {
            background: #ffffff;
            color: #1f2937;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            transition: transform 0.2s ease-in-out;
        }
        .card:hover {
            transform: translateY(-3px);
        }

        /* Score box */
        .score-box {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 8px;
            background-color: #1d4ed8;
            color: #ffffff;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 10px;
        }

        /* Links */
        a {
            color: #2563eb;
            text-decoration: none;
            font-weight: 600;
        }
        a:hover {
            text-decoration: underline;
            color: #1e40af;
        }
    </style>
""", unsafe_allow_html=True)


# ---------- App Title ----------
st.markdown('<div class="title-style">🔍 PubMed Semantic Search</div>', unsafe_allow_html=True)

# ---------- Sidebar (Email Input) ----------
with st.sidebar:
    st.subheader("📧 User Information")
    email = st.text_input("Enter your email (required for PubMed API)", placeholder="you@example.com")

# ---------- Pipeline ----------
pipeline = RAGPipeline()

# ---------- Input ----------
st.markdown("#### 💡 Ask your medical question below:")
question = st.text_input("Type your question here...", placeholder="e.g. What are the latest treatments for heart attack?")

# ---------- Buttons Always Visible ----------
col1, col2 = st.columns(2)
results = None
summary = None

with col1:
    if st.button("📑 Show Top Articles", use_container_width=True):
        if not email.strip():
            st.warning("⚠️ Please enter your email before searching.")
        elif question.strip():
            results = pipeline.query(question, topk=5)
        else:
            st.warning("⚠️ Please enter a question before searching.")

with col2:
    if st.button("📝 Generate Summary", use_container_width=True):
        if not email.strip():
            st.warning("⚠️ Please enter your email before generating summary.")
        elif question.strip():
            with st.spinner("⚡ Generating summary from top articles..."):
                summary = chatbot_answer(question, topk=5)
        else:
            st.warning("⚠️ Please enter a question before generating summary.")

# ---------- Results Section ----------
if results:
    st.markdown("---")
    st.markdown("### 📚 Most Relevant Articles")

    seen = set()
    unique_results = []
    for result in results:
        pmid = result['metadata'].get('pmid') or result['metadata'].get('url')
        if pmid not in seen:   # only add unique articles
            seen.add(pmid)
            unique_results.append(result)

    for result in unique_results:
        abstract = result['metadata'].get('abstract', '')
        short_abstract = (abstract[:700] + "...") if len(abstract) > 700 else abstract

        st.markdown(
            f"""
            <div class="card">
                <div class="score-box">Score: {result['score']:.4f}</div><br>
                <b>📌 Title:</b> {result['metadata'].get('title', '')}<br><br>
                <b>📖 Abstract:</b> {short_abstract}<br><br>
                <b>🔗 Link:</b> <a href="{result['metadata'].get('url', '')}" target="_blank">{result['metadata'].get('url', '')}</a>
            </div>
            """,
            unsafe_allow_html=True
        )

elif results is not None:
    st.warning("⚠️ No relevant passages found. Try a different query.")

# ---------- Summary Section ----------
if summary is not None:
    st.markdown("### 🧾 Summary from Top Articles")
    for line in summary.splitlines():
        if line.strip():
            st.markdown(line.strip())
