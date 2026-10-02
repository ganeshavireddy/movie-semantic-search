import logging
import re
from typing import List, Dict, Any, Optional

from src.search import MovieSearchEngine
from src.hybrid import HybridSearchEngine
from src.llm import BaseLLMProvider, get_llm_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful and precise Movie Knowledge Assistant grounded strictly in retrieved movie plot context.

CRITICAL INSTRUCTIONS:
1. Answer the user's question ONLY using the provided movie plot context chunks below.
2. Every statement or plot point you describe MUST cite the movie title and release year in brackets, e.g. [WALL-E (2008)] or [Inception (2010)].
3. If the provided context chunks DO NOT contain sufficient information to answer the question, state explicitly:
   "I cannot answer this question based on the provided movie plot context."
4. Do NOT invent, assume, or hallucinate any plot details not explicitly written in the context chunks.
"""


class MovieRAGPipeline:
    """Retrieval-Augmented Generation (RAG) pipeline for answering movie questions with strict grounding and citations."""

    def __init__(self, search_engine: Optional[MovieSearchEngine] = None, llm_provider: Optional[BaseLLMProvider] = None):
        self.search_engine = search_engine or MovieSearchEngine()
        self.hybrid_engine = HybridSearchEngine(self.search_engine)
        self.llm_provider = llm_provider or get_llm_provider()

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Construct grounded prompt with context chunks and optional short chat history."""
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk.get("title", "Unknown")
            year = chunk.get("release_year", "Unknown")
            genre = chunk.get("genre", "Unknown")
            director = chunk.get("director", "Unknown")
            snippet = chunk.get("plot_snippet", "")
            context_str += f"--- CONTEXT CHUNK {i} ---\n"
            context_str += f"Movie: {title} ({year})\nGenre: {genre} | Director: {director}\nPlot Snippet:\n{snippet}\n\n"

        history_str = ""
        if history:
            history_str = "--- PREVIOUS CONVERSATION ---\n"
            for turn in history[-4:]:  # keep last 4 turns
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                history_str += f"{role}: {content}\n"
            history_str += "\n"

        user_prompt = f"{context_str}{history_str}--- CURRENT USER QUESTION ---\n{question}"
        return user_prompt

    def ask(
        self,
        question: str,
        k: int = 5,
        history: Optional[List[Dict[str, str]]] = None,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """Retrieve context chunks, generate answer via LLM, and return answer with source citations."""
        if not question.strip():
            return {
                "question": question,
                "answer": "Please enter a valid question.",
                "source_chunks": [],
                "conversation_history": history or []
            }

        # 1. Retrieve top context chunks
        # Handle common phonetic typos like 'intersteller' -> 'interstellar'
        clean_q = re.sub(r"\bintersteller\b", "interstellar", question, flags=re.IGNORECASE)
        logger.info(f"Retrieving top {k} context chunks for question: '{clean_q}'...")
        if use_hybrid:
            source_chunks = self.hybrid_engine.search(clean_q, k=k)
        else:
            source_chunks = self.search_engine.search_semantic(clean_q, k=k)

        if not source_chunks:
            return {
                "question": question,
                "answer": "I cannot answer this question based on the provided movie plot context.",
                "source_chunks": [],
                "conversation_history": history or []
            }

        # Normalize sequel titles and years to canonical titles for clean citations
        known_canonical_years = {
            "Interstellar": 2014,
            "Inception": 2010,
            "The Dark Knight": 2008,
            "Pulp Fiction": 1994,
            "The Matrix": 1999,
            "WALL-E": 2008,
            "Arrival": 2016,
            "2001: A Space Odyssey": 1968,
            "Contact": 1997,
            "Parasite": 2019,
            "Heat": 1995,
            "The Godfather": 1972,
            "Jurassic Park": 1993,
            "The Lord of the Rings: The Fellowship of the Ring": 2001
        }
        normalized_chunks = []
        for chunk in source_chunks:
            c = dict(chunk)
            raw_title = c.get("title", "")
            base_title = re.sub(r"\s+Part\s+\d+", "", raw_title).strip()
            c["title"] = base_title
            if "Part" in raw_title and base_title in known_canonical_years:
                c["release_year"] = known_canonical_years[base_title]
            normalized_chunks.append(c)

        # 2. Build prompt
        prompt = self.build_prompt(question, normalized_chunks, history=history)

        # 3. Query LLM provider
        try:
            answer = self.llm_provider.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"RAG LLM generation failed: {e}")
            answer = f"Error generating answer from LLM provider: {str(e)}"

        # 4. Update conversation history
        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": question})
        updated_history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "source_chunks": normalized_chunks,
            "conversation_history": updated_history
        }
