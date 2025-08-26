import streamlit as st
from backend.rag_pipeline import RAGPipeline
from chatbot_ollama import chatbot_answer

st.title("PubMed Semantic Search")

pipeline = RAGPipeline()

question = st.text_input("Ask a medical question:")
results = None
summary = None
if question:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Top Articles"):
            results = pipeline.query(question, topk=5)
    with col2:
        if st.button("Generate Summary from Top Articles"):
            with st.spinner("Generating summary..."):
                summary = chatbot_answer(question, topk=5)

    if results:
        st.write("### Most Relevant Passages:")
        for result in results:
            st.write(f"**Score:** {result['score']:.4f}")
            st.write(f"**Title:** {result['metadata'].get('title', '')}")
            st.write(f"**Abstract:** {result['metadata'].get('abstract', '')}")
            st.write(f"**Link:** {result['metadata'].get('url', '')}")
            st.write("---")
    elif results is not None:
        st.warning("No relevant passages found. Please try a different query.")

# Always show summary section at the end
if summary is not None:
    st.write("### Summary from Top Articles:")
    st.write(summary)