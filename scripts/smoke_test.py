import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search import MovieSearchEngine
from src.hybrid import HybridSearchEngine
from src.rag import MovieRAGPipeline
from src.llm import BaseLLMProvider


class MockLLM(BaseLLMProvider):
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        if "WALL-E" in prompt:
            return "In WALL-E (2008), WALL-E is a lonely trash-compacting robot left on a deserted Earth who falls in love with the probe ship EVE [WALL-E (2008)]."
        elif "Heat" in prompt:
            return "In Heat (1995), Neil McCauley leads a high-stakes robbery crew pursued by LAPD detective Vincent Hanna [Heat (1995)]."
        elif "Matrix" in prompt:
            return "In The Matrix (1999), computer programmer Thomas Anderson (Neo) discovers that human reality is a simulated illusion controlled by intelligent machines [The Matrix (1999)]."
        return "I cannot answer this question based on the provided movie plot context."


def run_smoke_test():
    print("=== Running End-to-End System Smoke Test ===")
    search_engine = MovieSearchEngine()
    hybrid_engine = HybridSearchEngine(search_engine)
    rag_pipeline = MovieRAGPipeline(search_engine=search_engine, llm_provider=MockLLM())

    queries = [
        "a lonely robot cleans up an abandoned Earth",
        "heist crew betrayed by their leader",
        "hacker Thomas Anderson discovers reality is a simulated illusion"
    ]

    for q in queries:
        print(f"\n--------------------------------------------------")
        print(f"QUERY: '{q}'")
        print("--------------------------------------------------")
        
        results = hybrid_engine.search(q, k=2)
        print("TOP HYBRID SEARCH MATCHES:")
        for idx, res in enumerate(results, start=1):
            print(f"  {idx}. {res['title']} ({res['release_year']}) - RRF Score: {res['score']:.4f}")
            print(f"     Genre: {res['genre']} | Director: {res['director']}")
            print(f"     Snippet: {res['plot_snippet'][:120]}...")

        rag_res = rag_pipeline.ask(q, k=2)
        print("\nRAG ANSWER:")
        print(f"  {rag_res['answer']}")

    print("\n=== All 3 Smoke Test Queries Completed Successfully! ===")


if __name__ == "__main__":
    run_smoke_test()
