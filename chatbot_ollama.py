import requests
from backend.rag_pipeline import RAGPipeline

def get_top_articles_context(question, topk=5):
    pipeline = RAGPipeline()
    results = pipeline.query(question, topk=topk)
    context = ""
    for result in results:
        title = result["metadata"].get("title", "")
        abstract = result["metadata"].get("abstract", "")
        context += f"Title: {title}\nAbstract: {abstract}\n\n"
    return context

# Call Ollama (llama3) to generate a summary/answer
# Assumes Ollama is running locally with llama3 model pulled
# You can change the URL and model as needed

def ollama_generate_summary(question, context, model=None):
    # Only use abstracts for summary
    if not context.strip():
        return "No relevant articles found."
    lines = context.strip().split('\n')
    abstracts = [line.replace('Abstract:', '').strip() for line in lines if line.startswith('Abstract:') and line.strip() != 'Abstract:']
    # Concatenate abstracts and split into sentences
    import re
    all_text = ' '.join(abstracts)
    sentences = re.split(r'(?<=[.!?]) +', all_text)
    # Take up to 5 sentences for a concise summary
    summary = ' '.join(sentences[:5])
    return f"Concise summary (max 5 lines):\n\n{summary}"

# Example chatbot function
def chatbot_answer(question, topk=5):
    context = get_top_articles_context(question, topk)
    if not context.strip():
        return "No relevant articles found."
    summary = ollama_generate_summary(question, context)
    return summary

# Example usage:
if __name__ == "__main__":
    user_question = input("Ask a question: ")
    print("Generating answer...")
    print(chatbot_answer(user_question, topk=5))
