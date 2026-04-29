import os
import re
import fitz
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()

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

MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 100

RULE_HEADER = re.compile(r"^(RULE\s+\d+)\.\s*(.*)", re.MULTILINE)
SECTION_HEADER = re.compile(r"^(Sec\s+\d+)", re.MULTILINE)
LETTER_ITEM = re.compile(r"^\s*([A-Z])\.\s", re.MULTILINE)
NUMBERED_ITEM = re.compile(r"^\s*(\d+)\.\s", re.MULTILINE)


def extract_full_text(pdf_path: str) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if text:
            pages.append((page_num + 1, text))
    doc.close()
    return pages


def build_sections(full_text: str) -> list[dict]:
    """Split text into hierarchical sections, tracking Rule/Sec/Item context."""
    lines = full_text.split("\n")
    sections = []
    current_rule = ""
    current_sec = ""
    current_item = ""
    current_lines = []

    def flush():
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text and len(text) > MIN_CHUNK_SIZE:
                context_parts = [p for p in [current_rule, current_sec, current_item] if p]
                context = " > ".join(context_parts)
                sections.append({"context": context, "text": text})

    for line in lines:
        rule_match = RULE_HEADER.match(line.strip())
        sec_match = SECTION_HEADER.match(line.strip())

        if rule_match:
            flush()
            title = rule_match.group(2).strip()
            current_rule = f"{rule_match.group(1)}. {title}" if title else rule_match.group(1)
            current_sec = ""
            current_item = ""
            current_lines = [line]
        elif sec_match:
            flush()
            current_sec = sec_match.group(1)
            current_item = ""
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return sections


def chunk_section(section: dict) -> list[str]:
    """Break a section into chunks, prepending context to each."""
    context = section["context"]
    text = section["text"]
    prefix = f"[{context}]\n" if context else ""

    if len(prefix) + len(text) <= MAX_CHUNK_SIZE:
        return [prefix + text]

    chunks = []
    paragraphs = re.split(r"\n\s*\n", text)
    current_chunk = prefix

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 > MAX_CHUNK_SIZE and len(current_chunk) > len(prefix) + MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())
            current_chunk = prefix + para
        else:
            current_chunk += "\n" + para if current_chunk != prefix else prefix + para

    if current_chunk.strip() and len(current_chunk.strip()) > len(prefix):
        chunks.append(current_chunk.strip())

    return chunks


def chunk_supplementary(pages: list[tuple[int, str]]) -> list[dict]:
    """Chunk supplementary rules by numbered rule sections."""
    full_text = "\n\n".join(text for _, text in pages)
    supp_rule = re.compile(r"^(\d+\.\s+[A-Z])", re.MULTILINE)

    chunks = []
    splits = list(supp_rule.finditer(full_text))

    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
        section_text = full_text[start:end].strip()

        if len(section_text) < MIN_CHUNK_SIZE:
            continue

        rule_num = section_text.split(".")[0].strip()
        context = f"Supplementary Rule {rule_num}"

        if len(section_text) <= MAX_CHUNK_SIZE:
            chunks.append({"context": context, "text": f"[{context}]\n{section_text}"})
        else:
            paras = re.split(r"\n\s*\n", section_text)
            current = f"[{context}]\n"
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 > MAX_CHUNK_SIZE and len(current) > len(f"[{context}]\n") + MIN_CHUNK_SIZE:
                    chunks.append({"context": context, "text": current.strip()})
                    current = f"[{context}]\n{para}"
                else:
                    current += "\n" + para if not current.endswith("\n") else para
            if current.strip():
                chunks.append({"context": context, "text": current.strip()})

    return chunks


ENRICH_PROMPT = """You are indexing a youth softball league's supplementary rules for search. Given this rule text, write 2-3 short sentences describing what game SITUATIONS this rule applies to. Use common coaching language — how would a coach or parent describe this situation during a game?

For example, for a rule about runners being limited to one base per pitch on a steal, you might write:
"Applies when a runner tries to advance during a pitch. Covers stolen bases and how far runners can go. Relevant when asking about base stealing limits."

Rule text:
{rule_text}

Write ONLY the situation description, nothing else."""


def enrich_supplementary_chunk(rule_text: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": ENRICH_PROMPT.format(rule_text=rule_text),
        }],
    )
    return message.content[0].text.strip()


def find_page_for_position(pages: list[tuple[int, str]], full_text: str, position: int) -> int:
    """Find which page a character position in the concatenated text belongs to."""
    offset = 0
    for page_num, text in pages:
        end = offset + len(text) + 2  # +2 for the \n\n join
        if position < end:
            return page_num
        offset = end
    return pages[-1][0] if pages else 1


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
        pages = extract_full_text(pdf_path)
        print(f"  Extracted text from {len(pages)} pages")

        if source == "supplementary":
            chunks_with_context = chunk_supplementary(pages)
            print(f"  Enriching {len(chunks_with_context)} chunks with semantic descriptions...")
            for chunk_info in chunks_with_context:
                enrichment = enrich_supplementary_chunk(chunk_info["text"])
                enriched_text = f"[SITUATIONS: {enrichment}]\n{chunk_info['text']}"
                all_docs.append(enriched_text)
                all_metadatas.append({
                    "source": source,
                    "priority": priority,
                    "context": chunk_info["context"],
                    "filename": filename,
                })
                all_ids.append(f"{source}_c{chunk_id}")
                chunk_id += 1
                print(f"    Enriched: {chunk_info['context']}")
            print(f"  Created {len(chunks_with_context)} enriched chunks")
        else:
            full_text = "\n\n".join(text for _, text in pages)
            sections = build_sections(full_text)
            doc_chunks = 0

            for section in sections:
                section_chunks = chunk_section(section)
                for chunk_text in section_chunks:
                    all_docs.append(chunk_text)
                    all_metadatas.append({
                        "source": source,
                        "priority": priority,
                        "context": section.get("context", ""),
                        "filename": filename,
                    })
                    all_ids.append(f"{source}_c{chunk_id}")
                    chunk_id += 1
                    doc_chunks += 1

            print(f"  Created {doc_chunks} chunks")

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
