import os
from dotenv import load_dotenv

load_dotenv()

import chromadb
import anthropic

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
TOP_K = 10

SYSTEM_PROMPT = """You are a knowledgeable softball rules assistant for a 10U AA-League girls softball team. You answer questions about game rules, situations, and procedures.

You have access to two sources of rules:
1. **Supplementary Rules** (priority: high) — These are the AA-League-specific rules from the Wheaton Park District. They modify or override the core rules to suit this league.
2. **Core Rules** (priority: standard) — These are the official USSSA Fastpitch rules that serve as the baseline.

IMPORTANT: When supplementary rules and core rules conflict, the SUPPLEMENTARY rules take precedence. Always note when a supplementary rule overrides a core rule.

When answering:
- Be concise and direct — this will be read on a phone during a game.
- Cite which source (supplementary or core) your answer comes from.
- If the rules don't clearly address the situation, say so and offer the closest applicable rule.
- If you're unsure, say so rather than guessing.
"""


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("softball_rules")


def retrieve_chunks(question: str, n_results: int = TOP_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    chunks.sort(key=lambda c: (0 if c["metadata"]["priority"] == "high" else 1, c["distance"]))
    return chunks


def build_context(chunks: list[dict]) -> str:
    sections = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"]["source"].upper()
        page = chunk["metadata"]["page"]
        priority = chunk["metadata"]["priority"]
        sections.append(
            f"[Source {i}: {source} RULES (priority: {priority}), page {page}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(sections)


def ask(question: str) -> str:
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Based on the following rules excerpts, answer this question:

**Question:** {question}

---

{context}""",
            }
        ],
    )
    return message.content[0].text
