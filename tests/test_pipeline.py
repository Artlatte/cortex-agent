"""Tests for the end-to-end RAG pipeline."""

from cortex.config import CortexConfig
from cortex.rag import RAGPipeline


def _config() -> CortexConfig:
    return CortexConfig.default()


async def test_pipeline_ingest_search_stats_save_load(tmp_path) -> None:
    config = _config()
    pipeline = RAGPipeline(config)

    (tmp_path / "alpha.txt").write_text(
        "The capital of France is Paris.", encoding="utf-8"
    )
    (tmp_path / "beta.md").write_text(
        "机器学习是人工智能的一个重要分支。", encoding="utf-8"
    )
    (tmp_path / "gamma.txt").write_text(
        "Python is a popular programming language.", encoding="utf-8"
    )

    report = await pipeline.ingest([tmp_path])
    assert report.errors == []
    assert report.files == 3
    assert report.documents == 3
    assert report.chunks == 3

    result = await pipeline.search("机器学习")
    assert "机器学习" in result.hits[0].doc.page_content

    stats = pipeline.stats()
    assert stats["documents"] == 3
    assert stats["chunks"] == 3
    assert stats["dim"] == config.embedding.dim

    save_dir = tmp_path / "store"
    pipeline.save(save_dir)
    restored = await RAGPipeline.load(save_dir, config)
    result2 = await restored.search("机器学习")
    assert "机器学习" in result2.hits[0].doc.page_content
    assert restored.stats()["chunks"] == stats["chunks"]


async def test_pipeline_ingest_corrupt_file_records_error(tmp_path) -> None:
    config = _config()
    pipeline = RAGPipeline(config)

    (tmp_path / "good.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"garbage that is not a pdf")

    report = await pipeline.ingest([tmp_path])
    assert report.files == 2
    assert report.documents == 1
    assert report.chunks == 1
    assert len(report.errors) == 1
    assert "bad.pdf" in report.errors[0]
