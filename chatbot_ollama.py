from backend.rag_pipeline import RAGPipeline
import ollama  # Make sure you have ollama SDK installed

pipeline = RAGPipeline()

def get_top_articles_context(question, topk=5):
    results = pipeline.query(question, topk=topk)
    articles_text = ""
    for result in results:
        title = result["metadata"].get("title", "")
        abstract = result["metadata"].get("abstract", "")
        articles_text += f"Title: {title}\nAbstract: {abstract}\n\n"
    return articles_text

def chatbot_answer(question, topk=5, model="llama3.2"):
    # Get top articles context
    context = get_top_articles_context(question, topk)
    if not context.strip():
        return "No relevant articles found."

    # Create a system message (instructions) and user message (content)
    system_message = {
        "role": "system",
        "content": "You are a medical research assistant. Summarize the articles concisely, highlighting key findings and important points. Do not repeat text. Do not include the instructions in your output."
    }

    user_message = {
        "role": "user",
        "content": context
    }

    # Call Ollama
    response = ollama.chat(
        model=model,
        messages=[system_message, user_message]
    )

    # Extract only the text content
    summary = str(response).strip()  # or response.message.content if your SDK supports it

    return summary