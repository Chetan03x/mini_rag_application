import google.generativeai as genai
import os

# Load Gemini API key
GEN_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEN_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in your environment variables")

genai.configure(api_key=GEN_API_KEY)

def build_prompt_and_sources(query, top_items):
    """
    Build a Gemini prompt with numbered context for grounded answers.
    Returns both the prompt and formatted sources list.
    """
    context_blocks = []
    sources = []

    for i, item in enumerate(top_items, start=1):
        block = f"[{i}] {item['text']}"
        context_blocks.append(block)
        sources.append({
            "n": i,
            "title": item.get("title", "Untitled"),
            "snippet": item["text"][:200]
        })

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a factual assistant. Use ONLY the provided context to answer the question.
- Always cite sources inline with their numbers (e.g., [1], [2]).
- Do not use information outside the context.
- If the answer cannot be found in the context, reply: "I don't know based on the given sources."

Context:
{context}

Question: {query}

Answer (with inline citations):
"""

    return prompt, sources


def generate_answer_stream(query, top_items):
    """
    Generate Gemini answer with inline citations, streamed in chunks.
    """
    prompt, sources = build_prompt_and_sources(query, top_items)

    try:
        # Use Gemini 1.5 flash for fast streaming
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)

        # Get the grounded text with inline citations
        answer_text = response.text if hasattr(response, "text") else "I couldn't generate a response."

        # Stream chunks for smooth UI rendering
        for i in range(0, len(answer_text), 120):
            yield answer_text[i:i+120]

        # After answer, show the reference list
        yield "\n\n📚 Sources:\n" + "\n".join(
            f"[{s['n']}] {s['title']} — {s['snippet']}" for s in sources
        )

    except Exception as e:
        yield f"LLM call failed: {e}"