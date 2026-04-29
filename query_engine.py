import os
import re
from dotenv import load_dotenv

load_dotenv()

import chromadb
import anthropic

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
TOP_K = 15
EXPANDED_QUERIES = 3

SYSTEM_PROMPT = """You are a knowledgeable softball rules assistant for a 10U AA-League girls softball team. You answer questions about game rules, situations, and procedures.

You have access to two sources of rules:
1. **Supplementary Rules** (priority: high) — These are the AA-League-specific rules from the Wheaton Park District. They modify or override the core rules to suit this league.
2. **Core Rules** (priority: standard) — These are the official USSSA Fastpitch rules that serve as the baseline.

IMPORTANT: When supplementary rules and core rules conflict, the SUPPLEMENTARY rules take precedence. Always note when a supplementary rule overrides a core rule.

When answering:
- Start with a clear, direct YES or NO (or the key fact), then explain. Never contradict your opening line.
- Be concise and direct — this will be read on a phone during a game.
- Use minimal formatting. No markdown headers (#), horizontal rules (---), or excessive bullet nesting. Short paragraphs and bold for emphasis are fine.
- Cite which source (supplementary or core) your answer comes from.
- If the rules don't clearly address the situation, say so and offer the closest applicable rule.
- If you're unsure, say so rather than guessing.
"""

QUERY_EXPANSION_PROMPT = """You are helping search a softball rulebook. Given this question, generate {n} search queries that use the EXACT kind of language you'd find in an official rule book. Think about what words the rule itself would contain — not how a coach would ask the question.

For example, if the question is "Can you run on a dropped third strike?", good expansions would be:
- "charged with a third strike batter becomes runner"
- "third strike not caught before touching the ground"

Return ONLY the alternative queries, one per line, no numbering or bullets.

Question: {question}"""

KEYWORD_PROMPT = """Given this softball rules question, generate 3-5 short phrases (2-4 words each) that might appear VERBATIM in an official softball rulebook addressing this situation. Think about the exact wording a rulebook would use.

For example, for "If a pitch bounces before hitting the batter, does the batter get a base?":
- ball strikes the ground
- hit by a pitched ball
- batter hit by pitch
- awarded first base

For "Can a runner steal home?":
- steal home
- runner advance to home
- advancing to home plate

Return ONLY the phrases, one per line.

Question: {question}"""


def get_client():
    return anthropic.Anthropic()


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("softball_rules")


def expand_query(question: str) -> list[str]:
    client = get_client()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": QUERY_EXPANSION_PROMPT.format(n=EXPANDED_QUERIES, question=question),
        }],
    )
    expansions = [line.strip() for line in message.content[0].text.strip().split("\n") if line.strip()]
    return expansions[:EXPANDED_QUERIES]


def extract_keywords(question: str) -> list[str]:
    client = get_client()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": KEYWORD_PROMPT.format(question=question),
        }],
    )
    keywords = [line.strip() for line in message.content[0].text.strip().split("\n") if line.strip()]
    return keywords


def keyword_search(keywords: list[str], collection, n_results: int = 10) -> list[dict]:
    """Search all documents for keyword matches. Multi-word phrase matches score higher."""
    all_docs = collection.get(include=["documents", "metadatas"])
    scored = []

    for doc_id, doc_text, metadata in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
        doc_lower = doc_text.lower()
        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            phrase_matches = len(re.findall(re.escape(kw_lower), doc_lower))
            if phrase_matches > 0:
                word_count = len(kw_lower.split())
                score += phrase_matches * (word_count ** 2)
            else:
                words = kw_lower.split()
                if len(words) > 1:
                    matched_words = sum(1 for w in words if w in doc_lower)
                    if matched_words >= 2:
                        score += matched_words * 0.5

        if score > 0:
            scored.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": metadata,
                "keyword_score": score,
            })

    scored.sort(key=lambda x: -x["keyword_score"])
    return scored[:n_results]


SUPPLEMENTARY_SLOTS = 3


def retrieve_chunks(question: str, n_results: int = TOP_K) -> list[dict]:
    collection = get_collection()

    queries = [question] + expand_query(question)
    keywords = extract_keywords(question)

    seen_ids = set()
    all_chunks = []

    for query in queries:
        results = collection.query(query_texts=[query], n_results=n_results)
        for i in range(len(results["documents"][0])):
            chunk_id = results["ids"][0][i]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            all_chunks.append({
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "keyword_score": 0,
            })

    # Dedicated supplementary search — ensures the best supplementary
    # matches are always considered, even if they rank poorly globally
    for query in queries:
        supp_results = collection.query(
            query_texts=[query],
            n_results=SUPPLEMENTARY_SLOTS,
            where={"source": "supplementary"},
        )
        for i in range(len(supp_results["documents"][0])):
            chunk_id = supp_results["ids"][0][i]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append({
                    "id": chunk_id,
                    "text": supp_results["documents"][0][i],
                    "metadata": supp_results["metadatas"][0][i],
                    "distance": supp_results["distances"][0][i],
                    "keyword_score": 0,
                })

    kw_results = keyword_search(keywords, collection)
    for kw_chunk in kw_results:
        if kw_chunk["id"] in seen_ids:
            for c in all_chunks:
                if c["id"] == kw_chunk["id"]:
                    c["keyword_score"] = kw_chunk["keyword_score"]
                    break
        else:
            seen_ids.add(kw_chunk["id"])
            all_chunks.append({
                "id": kw_chunk["id"],
                "text": kw_chunk["text"],
                "metadata": kw_chunk["metadata"],
                "distance": 0.4,
                "keyword_score": kw_chunk["keyword_score"],
            })

    max_kw = max((c["keyword_score"] for c in all_chunks), default=1) or 1

    def score(chunk):
        priority = 0 if chunk["metadata"]["priority"] == "high" else 1
        vector_score = chunk["distance"]
        kw_bonus = (chunk["keyword_score"] / max_kw) * 0.3
        return (priority, vector_score - kw_bonus)

    all_chunks.sort(key=score)
    return all_chunks[:n_results]


def build_context(chunks: list[dict]) -> str:
    sections = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"]["source"].upper()
        priority = chunk["metadata"]["priority"]
        rule_context = chunk["metadata"].get("context", "")
        header = f"[Source {i}: {source} RULES (priority: {priority})"
        if rule_context:
            header += f", {rule_context}"
        header += "]"
        sections.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(sections)


def ask(question: str) -> str:
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    client = get_client()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
