"""test_rag.py: run the whole pipeline once in the terminal."""
from rag import build_messages, generate, index_pdf, retrieve

n_chunks = index_pdf("sample.pdf", "sample.pdf")
print(f"Indexed {n_chunks} chunks\n")

question = "What is this document about?"
hits = retrieve(question)

print("Retrieved chunks:")
for hit in hits:
    print(f"  page {hit['page']}  similarity {hit['score']}  {hit['text'][:70]}...")

print("\nAnswer:")
for piece in generate(build_messages(question, hits)):
    print(piece, end="", flush=True)
print()
