"""Deterministic baseline retrieval pipelines for thesis comparison.

Each baseline exposes a ``.run(query) -> answer dict`` shaped exactly like
the RLM pipeline's output, so :func:`akn_rlm.eval.runner._answer_to_result`
can convert them into metrics-ready records without any branching.
"""

from akn_rlm.baselines.bm25_pipeline import (
    BM25BaselinePipeline,
    build_bm25_pipeline,
)
from akn_rlm.baselines.dense_pipeline import (
    DenseBaselinePipeline,
    build_dense_pipeline,
)
from akn_rlm.baselines.hybrid_pipeline import (
    HybridBaselinePipeline,
    build_hybrid_pipeline,
)
from akn_rlm.baselines.hybrid_rerank_pipeline import (
    HybridRerankBaselinePipeline,
    build_hybrid_rerank_pipeline,
)
from akn_rlm.baselines.kg_pipeline import (
    KGBaselinePipeline,
    build_kg_pipeline,
)
from akn_rlm.baselines.kg_hybrid_pipeline import (
    KGHybridBaselinePipeline,
    build_kg_hybrid_pipeline,
)
from akn_rlm.baselines.llm_only_pipeline import (
    LLMOnlyBaselinePipeline,
    build_llm_only_pipeline,
)

__all__ = [
    "BM25BaselinePipeline",
    "build_bm25_pipeline",
    "DenseBaselinePipeline",
    "build_dense_pipeline",
    "HybridBaselinePipeline",
    "build_hybrid_pipeline",
    "HybridRerankBaselinePipeline",
    "build_hybrid_rerank_pipeline",
    "KGBaselinePipeline",
    "build_kg_pipeline",
    "KGHybridBaselinePipeline",
    "build_kg_hybrid_pipeline",
    "LLMOnlyBaselinePipeline",
    "build_llm_only_pipeline",
]
