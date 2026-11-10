"""app.py: a chat app for your PDF. Run with: streamlit run app.py"""
import streamlit as st

from rag import LLM_MODEL, build_messages, generate, get_collection, index_pdfs, retrieve

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption(f"Local RAG demo · embeddings: all-MiniLM-L6-v2 · LLM: {LLM_MODEL} (Ollama)")

# st.session_state keeps values between Streamlit's reruns (every click reruns this script)
if "messages" not in st.session_state:
    st.session_state.messages = []        # the chat history shown on screen
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []  # names of the PDFs currently in the database

# ---------- Sidebar: upload and index the PDF ----------
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )
    uploaded_names = [file.name for file in uploaded]

    if uploaded_names != st.session_state.indexed_files:
        if not uploaded_names:
            get_collection(reset=True)
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.rerun()
        with st.spinner("Reading, chunking and embedding..."):
            try:
                n_chunks = index_pdfs([(file, file.name) for file in uploaded])
            except ValueError as error:
                st.error(str(error))
                st.stop()
        st.session_state.indexed_files = uploaded_names
        st.session_state.messages = []    # new document, new conversation
        st.success(f"Indexed {n_chunks} chunks")

    if st.session_state.indexed_files:
        st.info("Chatting with: " + ", ".join(st.session_state.indexed_files))

# ---------- Main area: the chat ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF",
                         disabled=not st.session_state.indexed_files)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(generate(build_messages(question, hits)))
        with st.expander("Sources used"):
            for hit in hits:
                st.markdown(f"**Page {hit['page']}** · similarity {hit['score']}")
                st.caption(hit["text"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
