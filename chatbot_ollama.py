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

def chatbot_answer(question, topk=10, model="llama3.2"):
    # Get top articles context
    context = get_top_articles_context(question, topk)
    if not context.strip():
        return "No relevant articles found."

    # Create a system message (instructions) and user message (content)
    system_message = {
        "role": "system",
        "content": (
            "You are a medical research assistant. Read the following abstracts from top PubMed articles. "
            "Generate a single, concise summary that synthesizes the key findings, important points, and overall insights from these articles. "
            "Do not repeat the abstracts verbatim. Instead, combine the information and present it in your own words as a structured summary. "
            "Do not include these instructions in your output."
        )
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

    # Extract only the 'content' field from the message
    if hasattr(response, 'message') and hasattr(response.message, 'content'):
        summary = response.message.content.strip()
    elif isinstance(response, dict) and 'message' in response and 'content' in response['message']:
        summary = response['message']['content'].strip()
    else:
        # Fallback: try to extract content from string
        import re
        match = re.search(r"content='([^']+)'", str(response))
        summary = match.group(1).strip() if match else str(response).strip()

    return summary