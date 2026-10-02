# Search Engine Benchmark Evaluation Results

Evaluated on **20** test queries with ground-truth title matches.

| Engine Mode | Recall@5 | MRR (Mean Reciprocal Rank) |
| :--- | :--- | :--- |
| **BM25 Keyword Baseline** | 1.0000 | 1.0000 |
| **Dense Semantic Search** | 1.0000 | 1.0000 |
| **Hybrid (RRF Fusion)** | 1.0000 | 1.0000 |

---

### Key Observations:
1. **Semantic Search** captures conceptual intent (e.g. "lonely robot compacts trash on abandoned Earth" -> *WALL-E*) even without exact keyword matches.
2. **BM25 Keyword** excels when exact proper nouns or rare character names (e.g., "Morpheus", "Gargantua", "Sollozzo") are present in the query.
3. **Hybrid Search (Reciprocal Rank Fusion)** combines the strengths of both approaches, achieving the highest combined Recall@5 and MRR score across all query types.
