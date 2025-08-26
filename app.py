import streamlit as st
from backend.rag_pipeline import RAGPipeline

st.title("PubMed Semantic Search")

pipeline = RAGPipeline()

question = st.text_input("Ask a medical question:")
if question:
    results = pipeline.query(question, topk=5)
    if results:
        st.write("### Most Relevant Passages:")
        for result in results:
            st.write(f"**Score:** {result['score']:.4f}")
            st.write(f"**Title:** {result['metadata'].get('title', '')}")
            st.write(f"**Abstract:** {result['metadata'].get('abstract', '')}")
            st.write("---")
    else:
        st.warning("No relevant passages found. Please try a different query.")