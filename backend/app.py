import streamlit as st
import time
from embeddings import extract_text_from_file, chunk_text, embed_texts
from vectorstore import init_vectorstore, upsert_points, search_vectors
from retriever import retrieve
from llm_answer import build_prompt_and_sources
from config import TOPK_VECTOR, TARGET_TOKENS, OVERLAP_TOKENS, GEMINI_API_KEY
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="MiniRAG-Gemini", layout="wide")
st.title("MiniRAG Application")

# Sidebar
with st.sidebar.expander("Settings & Keys", expanded=False):
    st.write("Make sure you set environment variables or fill `.env` before running.")
    st.markdown("- `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `COHERE_API_KEY`")
    st.write(f"Chunk target tokens: {TARGET_TOKENS}, overlap: {OVERLAP_TOKENS}")

# Ensure collection
if st.sidebar.button("Ensure Qdrant collection"):
    try:
        _ = init_vectorstore()
        st.sidebar.success("Qdrant collection ensured.")
    except Exception as e:
        st.sidebar.error(f"Could not ensure collection: {e}")

# 1) Add content
st.header("Add content")
col1, col2 = st.columns([3, 1])
with col1:
    title = st.text_input("Document title", "Untitled Document")
    source = st.text_input("Source label / URL (optional)", "user-paste")
    text_area = st.text_area("Paste text here", height=200)
    uploaded_file = st.file_uploader("Or upload a file (.txt, .md, .pdf)", type=["txt", "md", "pdf"])

with col2:
    if st.button("Index content"):
        with st.spinner("Indexing..."):
            try:
                raw = text_area or ""
                if uploaded_file:
                    raw += "\n" + extract_text_from_file(uploaded_file)
                if not raw.strip():
                    st.error("No text provided.")
                else:
                    # ✅ chunk_text only takes raw text
                    chunks = chunk_text(raw)

                    texts = [c["text"] for c in chunks]
                    vectors = embed_texts(texts)

                    ids = [c["id"] for c in chunks]
                    payloads = [
                        {"title": title, "source": source, **c["meta"], "text": c["text"]}
                        for c in chunks
                    ]

                    # ✅ upsert_points expects separate args
                    upsert_points(ids, vectors, payloads)

                    st.success(f"Indexed {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

st.markdown("---")

# 2) Ask question
st.header("Ask a question")
query = st.text_input("Your question", "")

colA, colB = st.columns([3, 1])  # wider left panel for answer

with colB:
    ask_button = st.button("Ask")

if ask_button:
    if not query.strip():
        st.warning("Enter a question.")
    else:
        start = time.time()
        try:
            qvec = embed_texts([query])[0]
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            qvec = None

        if qvec:
            try:
                hits = search_vectors(qvec, top_k=TOPK_VECTOR, with_payload=True)
                if not hits:
                    st.info("I couldn't find this in your data.")
                else:
                    top_items = []
                    for hit in hits:
                        payload = hit.payload
                        text = payload.get("text", "")
                        t = payload.get("title", "Untitled")
                        s = payload.get("source", "unknown")
                        if text:
                            top_items.append({"text": text, "title": t, "source": s})

                    if not top_items:
                        st.info("No valid text found in retrieved items.")
                    else:
                        # Two-panel layout
                        ans_col, meta_col = st.columns([2.5, 1])

                        with ans_col:
                            st.markdown("### 💡 Answer")
                            answer_container = st.empty()
                            partial_text = ""

                        with meta_col:
                            st.markdown("### 📂 Sources")
                            sources_box = st.empty()
                            st.markdown("### ⚙️ Diagnostics")
                            diag_box = st.empty()

                        # Build prompt
                        prompt, sources = build_prompt_and_sources(query, top_items)

                        try:
                            response = gemini_model.generate_content(prompt, stream=True)

                            for chunk in response:
                                if chunk.text:
                                    partial_text += chunk.text
                                    # ✅ Answer card style
                                    answer_container.markdown(
                                        f"""
                                        <div style="padding:15px; border-radius:10px; 
                                        border:1px solid #444; background-color:#1e1e1e; 
                                        box-shadow: 0px 2px 6px rgba(0,0,0,0.3);">
                                        {partial_text}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # ✅ Sources card (only once)
                                    if sources and sources_box:
                                        src_txt = ""
                                        for s in sources:
                                            src_txt += f"[{s['n']}] {s['title']} — {s['snippet']}...\n"
                                        sources_box.markdown(
                                            f"""
                                            <div style="padding:10px; border-radius:10px; 
                                            border:1px solid #555; background-color:#252525;">
                                            {src_txt}
                                    
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                        sources = None

                            # End of answer footer
                            answer_container.markdown(
                                f"""
                                <div style="padding:15px; border-radius:10px; 
                                border:1px solid #444; background-color:#1e1e1e; 
                                box-shadow: 0px 2px 6px rgba(0,0,0,0.3);">
                                {partial_text}
                                
                                
                                """,
                                unsafe_allow_html=True,
                            )

                        except Exception as e:
                            st.error(f"LLM call failed: {e}")

                        # ✅ Diagnostics card
                        diag_box.markdown(
                            f"""
                            <div style="padding:10px; border-radius:10px; 
                            border:1px solid #555; background-color:#252525;">
                            <b>Elapsed seconds:</b> {round(time.time() - start, 2)}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Query pipeline failed: {e}")

