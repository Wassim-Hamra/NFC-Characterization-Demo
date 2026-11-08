"""
rag_engine.py
-------------
BM25-based document retrieval + optional Gemini API answer synthesis.

The retrieval runs entirely locally (pure Python BM25, no downloads).
Gemini is optional: without a GOOGLE_API_KEY the engine falls back to
displaying retrieved passages directly, clearly labelled as retrieval-only.
"""

import os
import re
import math
import json
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import certifi

load_dotenv()

# Ensure SSL certificates are properly resolved on Windows
os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

DOCS_PATH = Path("data/sample_documents")

PRIMARY_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]

SYSTEM_PROMPT = """You are an expert NFC semiconductor characterisation engineer assistant.
Answer the engineer's question using ONLY the provided context passages from the technical documentation.
Do not invent specification limits or claim confirmed physical causes without evidence from the context.
If the context does not contain enough information to answer the question, say so explicitly.
Cite the document name when referencing information.
Distinguish between retrieved facts and possible explanations.
Keep the answer concise and technically accurate."""


# ─────────────────────────────────────────────────────────────────
# Document model
# ─────────────────────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    source_file: str


@dataclass
class SearchResult:
    document: Document
    score: float
    excerpt: str    # most relevant ~200 chars


# ─────────────────────────────────────────────────────────────────
# BM25 Index
# ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """Lightweight BM25 retrieval index — no external dependencies."""

    K1 = 1.5
    B  = 0.75

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.n = len(documents)
        self._tokenized = [_tokenize(d.content) for d in documents]
        self._doc_lengths = [len(t) for t in self._tokenized]
        self._avgdl = sum(self._doc_lengths) / max(self.n, 1)
        self._idf = self._compute_idf()
        self._tf  = self._compute_tf()

    def _compute_idf(self) -> dict[str, float]:
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def _compute_tf(self) -> list[dict[str, int]]:
        tfs = []
        for tokens in self._tokenized:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            tfs.append(tf)
        return tfs

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        dl = self._doc_lengths[doc_idx]
        tf = self._tf[doc_idx]
        score = 0.0
        for t in query_tokens:
            if t not in self._idf:
                continue
            f = tf.get(t, 0)
            numerator   = f * (self.K1 + 1)
            denominator = f + self.K1 * (1 - self.B + self.B * dl / self._avgdl)
            score += self._idf[t] * numerator / (denominator + 1e-9)
        return score

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        query_tokens = _tokenize(query)
        scores = [(i, self._score(query_tokens, i)) for i in range(self.n)]
        scores.sort(key=lambda x: -x[1])
        results = []
        for idx, score in scores[:k]:
            if score < 0.01:
                break
            doc = self.documents[idx]
            excerpt = _extract_excerpt(doc.content, query_tokens)
            results.append(SearchResult(document=doc, score=score, excerpt=excerpt))
        return results


def _extract_excerpt(content: str, query_tokens: list[str], window: int = 300) -> str:
    """Find the most relevant excerpt in a document for the given query tokens."""
    lower = content.lower()
    best_pos = 0
    best_hits = 0
    for i in range(0, max(1, len(lower) - window), 50):
        chunk = lower[i : i + window]
        hits = sum(1 for t in query_tokens if t in chunk)
        if hits > best_hits:
            best_hits = hits
            best_pos = i
    return content[best_pos : best_pos + window].strip().replace("\n", " ")


# ─────────────────────────────────────────────────────────────────
# Document loader
# ─────────────────────────────────────────────────────────────────

def load_documents(docs_path: Path = DOCS_PATH) -> list[Document]:
    documents = []
    for path in sorted(docs_path.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        # Extract title from first non-empty line after the disclaimer header
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        title = lines[1] if len(lines) > 1 else path.stem
        documents.append(Document(
            doc_id=path.stem,
            title=title.replace("Document: ", "").strip(),
            content=content,
            source_file=path.name,
        ))
    return documents


# ─────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self.documents = load_documents()
        self.index = BM25Index(self.documents)
        self._gemini_client = None
        self._gemini_available = False
        self._model_name = PRIMARY_MODELS[0]
        self._init_gemini()

    def _init_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            try:
                import streamlit as st
                api_key = str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
            except Exception:
                api_key = ""
        if not api_key:
            return
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            self._model_name = PRIMARY_MODELS[0]
            self._gemini_available = True
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=api_key, transport="rest")
                self._gemini_client = legacy_genai.GenerativeModel(PRIMARY_MODELS[0])
                self._gemini_available = True
            except Exception:
                self._gemini_available = False

    @property
    def llm_available(self) -> bool:
        return self._gemini_available

    def retrieve(self, question: str, k: int = 3) -> list[SearchResult]:
        return self.index.search(question, k=k)

    def answer(
        self,
        question: str,
        extra_context: str = "",
        k: int = 3,
    ) -> dict:
        """
        Retrieve relevant passages and generate an LLM answer.
        """
        retrieved = self.retrieve(question, k=k)

        if not retrieved:
            return {
                "question": question,
                "retrieved": [],
                "answer": None,
                "mode": "retrieval_only",
                "error": "No relevant passages found in the documentation.",
            }

        if not self._gemini_available:
            return {
                "question": question,
                "retrieved": retrieved,
                "answer": None,
                "mode": "retrieval_only",
                "error": None,
            }

        # Build context for the LLM
        passages = "\n\n".join([
            f"[{r.document.title}]\n{r.document.content[:3500]}"
            for r in retrieved
        ])
        if extra_context:
            passages = f"[ANALYSIS CONTEXT]\n{extra_context}\n\n{passages}"

        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{passages}\n\nQuestion: {question}\nAnswer:"

        # Try generation with candidate models
        last_error = None
        for model_candidate in PRIMARY_MODELS:
            try:
                if hasattr(self._gemini_client, "models"):
                    # google-genai SDK
                    response = self._gemini_client.models.generate_content(
                        model=model_candidate,
                        contents=prompt,
                    )
                    answer_text = response.text.strip()
                else:
                    # legacy google-generativeai SDK
                    response = self._gemini_client.generate_content(prompt)
                    answer_text = response.text.strip()

                return {
                    "question": question,
                    "retrieved": retrieved,
                    "answer": answer_text,
                    "mode": "llm",
                    "error": None,
                }
            except Exception as e:
                last_error = str(e)
                continue

        return {
            "question": question,
            "retrieved": retrieved,
            "answer": None,
            "mode": "retrieval_only",
            "error": f"LLM generation unavailable: {last_error[:200] if last_error else 'Unknown error'}",
        }
