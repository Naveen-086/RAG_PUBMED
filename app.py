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
        body {
            background-color: #f5f7fa; /* Light background */
            color: #2C3E50; /* Dark text */
        }
        .title-style {
            font-size: 38px;
            font-weight: 700;
            color: #2C3E50;
            text-align: center;
            padding: 10px;
        }
        .stTextInput input {
            border-radius: 12px;
            border: 1px solid #3498db;
            padding: 10px;
        }
        .card {
            background-color: #ffffff; /* White background for cards */
            color: #2C3E50; /* Dark readable text */
            padding: 18px;
            border-radius: 12px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 15px;
        }
        .score-box {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            background-color: #3498db;
            color: #ffffff; /* White text on blue */
            font-weight: bold;
            font-size: 13px;
            margin-bottom: 5px;
        }
        a {
            color: #2980b9;
            text-decoration: none;
            font-weight: 600;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- App Title ----------
st.markdown('<div class="title-style">🔍 PubMed Semantic Search</div>', unsafe_allow_html=True)

# ---------- Pipeline ----------
pipeline = RAGPipeline()

# ---------- Input ----------
st.markdown("#### 💡 Ask your medical question below:")
question = st.text_input("Type your question here...", placeholder="e.g. What are the latest treatments for heart attack?")
results = None
summary = None

if question:
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📑 Show Top Articles", use_container_width=True):
            results = pipeline.query(question, topk=5)

    with col2:
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("⚡ Generating summary from top articles..."):
                summary = chatbot_answer(question, topk=5)

    # ---------- Results Section ----------
        # ---------- Results Section ----------
    if results:
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
    st.success(summary)