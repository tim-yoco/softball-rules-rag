import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

import anthropic
from vector_store import query_store, load_store

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


def normalize_text(text: str) -> str:
    """Normalize smart quotes and other unicode variants for matching."""
    return (text
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))


def keyword_search(keywords: list[str], n_results: int = 10) -> list[dict]:
    """Search all documents for keyword matches. Multi-word phrase matches score higher."""
    _, metadata = load_store()
    scored = []

    for i, entry in enumerate(metadata):
        doc_lower = normalize_text(entry["text"]).lower()
        score = 0.0
        for kw in keywords:
            kw_lower = normalize_text(kw).lower()
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
                "id": f"{entry['source']}_c{i}",
                "text": entry["text"],
                "metadata": entry,
                "keyword_score": score,
            })

    scored.sort(key=lambda x: -x["keyword_score"])
    return scored[:n_results]


SUPPLEMENTARY_SLOTS = 3


def retrieve_chunks(question: str, n_results: int = TOP_K) -> list[dict]:
    queries = [question] + expand_query(question)
    keywords = extract_keywords(question)

    seen_ids = set()
    all_chunks = []

    for query in queries:
        results = query_store([query], n_results=n_results)
        for chunk in results[0]:
            if chunk["id"] in seen_ids:
                continue
            seen_ids.add(chunk["id"])
            chunk["keyword_score"] = 0
            all_chunks.append(chunk)

    # Dedicated supplementary search
    for query in queries:
        supp_results = query_store([query], n_results=SUPPLEMENTARY_SLOTS,
                                   where={"source": "supplementary"})
        for chunk in supp_results[0]:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                chunk["keyword_score"] = 0
                all_chunks.append(chunk)

    kw_results = keyword_search(keywords)
    for kw_chunk in kw_results:
        if kw_chunk["id"] in seen_ids:
            for c in all_chunks:
                if c["id"] == kw_chunk["id"]:
                    c["keyword_score"] = kw_chunk["keyword_score"]
                    break
        else:
            seen_ids.add(kw_chunk["id"])
            kw_chunk["distance"] = 0.4
            all_chunks.append(kw_chunk)

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
