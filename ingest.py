import os
import fitz
import chromadb
from chromadb.config import Settings

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(DOCS_DIR, "chroma_db")

DOCUMENTS = [
    {
        "path": os.path.join(DOCS_DIR, "AA-League-Softball-Supplementary-Rules.pdf"),
        "source": "supplementary",
        "priority": "high",
    },
    {
        "path": os.path.join(DOCS_DIR, "Fastpitch_Rules.pdf"),
        "source": "core",
        "priority": "standard",
    },
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def extract_text_by_page(pdf_path: str) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if text:
            pages.append((page_num + 1, text))
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def ingest():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection("softball_rules")
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="softball_rules",
        metadata={"hnsw:space": "cosine"},
    )

    all_docs = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0

    for doc_info in DOCUMENTS:
        pdf_path = doc_info["path"]
        source = doc_info["source"]
        priority = doc_info["priority"]
        filename = os.path.basename(pdf_path)

        print(f"\nProcessing: {filename} (source={source}, priority={priority})")
        pages = extract_text_by_page(pdf_path)
        print(f"  Extracted text from {len(pages)} pages")

        doc_chunks = 0
        for page_num, page_text in pages:
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                all_docs.append(chunk)
                all_metadatas.append({
                    "source": source,
                    "priority": priority,
                    "page": page_num,
                    "filename": filename,
                })
                all_ids.append(f"{source}_p{page_num}_c{chunk_id}")
                chunk_id += 1
                doc_chunks += 1

        print(f"  Created {doc_chunks} chunks")

    # ChromaDB batches max 5461 items per add
    batch_size = 5000
    for i in range(0, len(all_docs), batch_size):
        end = min(i + batch_size, len(all_docs))
        collection.add(
            documents=all_docs[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end],
        )

    print(f"\nTotal chunks ingested: {collection.count()}")
    print(f"ChromaDB stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
