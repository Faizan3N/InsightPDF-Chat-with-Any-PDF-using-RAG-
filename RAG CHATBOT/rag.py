"""rag.py: the RAG pipeline. Load -> chunk -> embed -> store -> retrieve -> generate."""
import chromadb
import ollama
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Settings you can change ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"   # must match a model you pulled with Ollama
CHUNK_SIZE = 150            # words per chunk
CHUNK_OVERLAP = 30          # words shared by neighbouring chunks
TOP_K = 4                   # chunks retrieved per question
DB_DIR = "chroma_db"        # folder where ChromaDB saves its data
COLLECTION_NAME = "pdf_chunks"


# ---------- Step 3: load the PDF ----------
def load_pdf(file):
    """Read a PDF (a file path or an uploaded file) and return its pages:
    [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]"""
    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip empty pages
            pages.append({"page": number, "text": text})
    return pages


# ---------- Step 4: chunk ----------
def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split every page into overlapping chunks of words, remembering the page number."""
    step = chunk_size - overlap
    chunks = []
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            chunks.append({"text": piece, "page": page["page"]})
            if start + chunk_size >= len(words):  # this chunk reached the end of the page
                break
    return chunks


# ---------- Step 5: embed ----------
_embedder = None


def get_embedder():
    """Load the embedding model once, then reuse it (loading takes a few seconds)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed(texts):
    """Turn a list of texts into a list of vectors (each a list of 384 numbers, length 1)."""
    vectors = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ---------- Step 6: store ----------
def get_collection(reset=False):
    """Open the ChromaDB collection saved in DB_DIR. reset=True empties it first."""
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete yet
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def index_pdf(file, source_name):
    """The whole indexing pipeline: PDF -> pages -> chunks -> vectors -> ChromaDB.
    Replaces any previously indexed PDF. Returns the number of chunks stored."""
    return index_pdfs([(file, source_name)])


def index_pdfs(files):
    """Index multiple PDFs together, replacing the previous collection once."""
    all_chunks = []
    for file_number, (file, source_name) in enumerate(files):
        pages = load_pdf(file)
        chunks = chunk_pages(pages)
        for chunk_number, chunk in enumerate(chunks):
            chunk["source"] = source_name
            chunk["id"] = f"{file_number}-{source_name}-{chunk_number}"
        all_chunks.extend(chunks)

    chunks = all_chunks
    if not chunks:
        raise ValueError("No text found in this PDF. It may be a scanned image (see Troubleshooting).")
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    vectors = embed(texts)
    collection = get_collection(reset=True)
    batch = 500  # ChromaDB limits how many items one add() call can take
    for start in range(0, len(chunks), batch):
        end = start + batch
        collection.add(ids=ids[start:end], embeddings=vectors[start:end],
                       documents=texts[start:end], metadatas=metadatas[start:end])
    return len(chunks)


# ---------- Step 7: retrieve ----------
def retrieve(question, k=TOP_K):
    """Return the k chunks most similar to the question, best first."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=embed([question]),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "page": meta["page"], "source": meta["source"],
                     "score": round(1 - distance, 3)})  # cosine similarity = 1 - distance
    return hits


# ---------- Step 8: prompt + generate ----------
SYSTEM_PROMPT = """You answer questions about a PDF document.
Use ONLY the numbered context passages from the document.
If the answer is not in the passages, say: "I couldn't find that in the document."
After each fact, cite its page like this: (page 3).
Keep answers clear and short."""


def build_messages(question, hits):
    """Build the chat messages: rules, then the retrieved passages and the question."""
    context = "\n\n".join(f"[{i}] (page {h['page']})\n{h['text']}" for i, h in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def generate(messages):
    """Stream the answer from the local LLM, a few words at a time."""
    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True,
                         options={"temperature": 0.1})
    for part in stream:
        yield part["message"]["content"]
