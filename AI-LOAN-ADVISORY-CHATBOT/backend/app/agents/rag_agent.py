"""
RAG Agent — Phase 5
Generates answers strictly grounded in retrieved policy document evidence.

Design Principles:
  • Uses retrieved chunks (from Phase 4 vector_store) as the ONLY knowledge source.
  • Temperature is set to 0.1 — deterministic, no creative invention.
  • If the answer cannot be found in the evidence, returns NOT_IN_EVIDENCE signal.
  • Extracts which evidence blocks were cited so sources are accurate.
  • Never invents loan limits, rates, DTI thresholds, or eligibility rules.
"""

import os
import re
from typing import List, Dict, Optional

from google import genai
from google.genai import types

_client: Optional[genai.Client] = None
MODEL = "models/gemini-flash-latest"

# Sentinel returned by the model when evidence doesn't contain the answer
_NOT_IN_EVIDENCE = "NOT_IN_EVIDENCE"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _build_evidence_block(chunks: List[Dict]) -> str:
    """Format retrieved chunks as numbered evidence blocks for the prompt."""
    lines = []
    for i, ch in enumerate(chunks, 1):
        lines.append(f"[EVIDENCE {i}]")
        lines.append(f"  Document : {ch['document_name']}")
        lines.append(f"  Page     : {ch['page_number']}")
        if ch.get("section"):
            lines.append(f"  Section  : {ch['section']}")
        lines.append(f"  Text     :\n  {ch['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


def _extract_cited_indices(answer_text: str, num_chunks: int) -> List[int]:
    """
    Find which Evidence numbers were referenced in the answer text.
    Returns a sorted list of 0-based indices.
    """
    indices = set()
    for m in re.finditer(r'\bevidence\s+(\d+)\b', answer_text, re.IGNORECASE):
        idx = int(m.group(1)) - 1          # convert to 0-based
        if 0 <= idx < num_chunks:
            indices.add(idx)
    # Also look for "(Evidence N)" or "[Evidence N]" patterns
    for m in re.finditer(r'[\[\(]Evidence\s+(\d+)[\]\)]', answer_text, re.IGNORECASE):
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_chunks:
            indices.add(idx)
    return sorted(indices)


def generate_answer(question: str, chunks: List[Dict]) -> Dict:
    """
    Generate a strictly grounded answer from retrieved evidence.

    Args:
        question : The user's natural-language question.
        chunks   : Retrieved chunks from vector_store.search() with full metadata.

    Returns a dict with:
        answer          : The generated answer text, or None if not in evidence.
        not_in_evidence : True if the model could not find the answer.
        sources         : List of cited source metadata dicts.
        raw_response    : Raw LLM output string (for validation agent).
    """
    if not chunks:
        return {
            "answer":          None,
            "not_in_evidence": True,
            "sources":         [],
            "raw_response":    None,
        }

    evidence_block = _build_evidence_block(chunks)

    prompt = f"""You are a policy assistant for Tata Mitra, a loan advisory system.

Your task: Answer the user's question using the numbered evidence blocks below.

━━━━━━━━ RULES ━━━━━━━━
1. Use ONLY information from the provided evidence. Do NOT use your own general knowledge.
2. Never invent or assume any numbers: no loan limits, interest rates, credit scores,
   DTI ratios, income thresholds, tenure limits, or processing fees unless they appear
   verbatim in the evidence.
3. When you state a fact, reference its evidence block: e.g., "(Evidence 2)".
4. If the evidence contains PARTIAL information, answer with what IS available and note
   that full details may require consulting the bank directly.
5. ONLY respond with exactly the word {_NOT_IN_EVIDENCE} if the evidence blocks are
   completely unrelated to the question (zero relevant content). If there is any
   relevant partial information, use it to form a partial answer instead.
6. Be clear and direct. Do not pad or speculate.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVIDENCE:
{evidence_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER QUESTION: {question}

ANSWER (cite Evidence numbers for every fact you state):"""

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=800,
            ),
        )
    except Exception as api_err:
        print(f"RAGAgent Gemini API error: {api_err}")
        # Return a structured fallback so the caller can use the raw chunks
        # instead of crashing the entire request pipeline.
        return {
            "answer":          None,
            "not_in_evidence": True,
            "sources":         [],
            "raw_response":    None,
        }

    raw_answer = (response.text or "").strip()

    # Detect NOT_IN_EVIDENCE sentinel
    if raw_answer.upper().startswith(_NOT_IN_EVIDENCE) or raw_answer.strip() == _NOT_IN_EVIDENCE:
        return {
            "answer":          None,
            "not_in_evidence": True,
            "sources":         [],
            "raw_response":    raw_answer,
        }

    # Build sources from cited evidence blocks
    cited_indices = _extract_cited_indices(raw_answer, len(chunks))
    if not cited_indices:
        # Fallback: attribute top-2 most similar chunks
        cited_indices = list(range(min(2, len(chunks))))

    sources = [
        {
            "document_name":    chunks[i]["document_name"],
            "document_id":      chunks[i]["document_id"],
            "page_number":      chunks[i]["page_number"],
            "section":          chunks[i].get("section"),
            "chunk_id":         chunks[i]["chunk_id"],
            "relevance_score":  round(float(chunks[i].get("score", 0.0)), 4),
        }
        for i in cited_indices
    ]

    return {
        "answer":          raw_answer,
        "not_in_evidence": False,
        "sources":         sources,
        "raw_response":    raw_answer,
    }
