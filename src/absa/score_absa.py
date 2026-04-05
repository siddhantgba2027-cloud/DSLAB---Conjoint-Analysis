from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SCALE = ["worst", "bad", "okay", "good", "excellent"]
SCALE_MAP = {
    "positive": ["good", "excellent"],
    "negative": ["bad", "worst"],
    "neutral": ["okay"],
}
LABEL_TO_SCORE = {"worst": 1, "bad": 2, "okay": 3, "good": 4, "excellent": 5}


@dataclass(frozen=True)
class ScorePaths:
    combined_input: Path
    scored_output: Path
    aspect_summary_output: Path
    review_summary_output: Path


def default_paths(project_root: Path) -> ScorePaths:
    return ScorePaths(
        combined_input=project_root / "data" / "interim" / "absa_output_electric_tb_combined.csv",
        scored_output=project_root / "data" / "outputs" / "absa" / "final_bucket_output.csv",
        aspect_summary_output=project_root / "data" / "outputs" / "absa" / "aspect_summary.csv",
        review_summary_output=project_root / "data" / "outputs" / "absa" / "review_summary.csv",
    )


def build_text(row: pd.Series) -> str:
    aspect = str(row.get("aspect", "")).strip()
    opinion = str(row.get("opinion_term", "")).strip()
    return f"{aspect} {opinion}".strip()


def mean_embedding(model: object, text: str, vector_size: int) -> np.ndarray:
    words = str(text).lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def choose_bucket(
    text: str,
    sentiment: str,
    embed_text: Callable[[str], np.ndarray],
    scale_embeddings_by_sentiment: dict[str, np.ndarray],
) -> str:
    words = SCALE_MAP.get(str(sentiment).lower(), SCALE)
    scale_embeddings = scale_embeddings_by_sentiment.get(str(sentiment).lower())
    if scale_embeddings is None:
        scale_embeddings = scale_embeddings_by_sentiment["default"]
    text_embedding = embed_text(text).reshape(1, -1)
    similarities = cosine_similarity(text_embedding, scale_embeddings)
    return words[int(np.argmax(similarities))]


def summarize_by_aspect(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("aspect", dropna=False)
        .agg(
            mentions=("aspect", "size"),
            mean_avg_score=("avg_score", "mean"),
            dominant_sentiment=("sentiment", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
            sample_opinion=("opinion_term", lambda x: next((str(v) for v in x if pd.notna(v) and str(v).strip()), "")),
        )
        .reset_index()
        .sort_values(["mean_avg_score", "mentions"], ascending=[False, False])
    )
    summary["mean_avg_score"] = summary["mean_avg_score"].round(4)
    return summary


def summarize_by_review(df: pd.DataFrame) -> pd.DataFrame:
    review_summary = (
        df.groupby(["review_id", "asin", "star_rating"], dropna=False)
        .agg(
            extracted_aspects=("aspect", "size"),
            mean_avg_score=("avg_score", "mean"),
            unique_aspects=("aspect", lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v) and str(v).strip()}))),
        )
        .reset_index()
        .sort_values("mean_avg_score", ascending=False)
    )
    review_summary["mean_avg_score"] = review_summary["mean_avg_score"].round(4)
    return review_summary


def score_absa(paths: ScorePaths) -> dict[str, pd.DataFrame]:
    try:
        import gensim.downloader as api
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "ABSA dependencies are missing. Run `pip install -r requirements.txt` from the project root "
            "and then rerun `python run_absa.py`."
        ) from exc

    df = pd.read_csv(paths.combined_input).copy()
    df["text_for_analysis"] = df.apply(build_text, axis=1)

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    w2v_model = api.load("word2vec-google-news-300")
    glove_model = api.load("glove-wiki-gigaword-300")
    sbert_scale_embeddings = {
        sentiment: sbert_model.encode(words)
        for sentiment, words in {**SCALE_MAP, "default": SCALE}.items()
    }
    w2v_scale_embeddings = {
        sentiment: np.array([mean_embedding(w2v_model, word, 300) for word in words])
        for sentiment, words in {**SCALE_MAP, "default": SCALE}.items()
    }
    glove_scale_embeddings = {
        sentiment: np.array([mean_embedding(glove_model, word, 300) for word in words])
        for sentiment, words in {**SCALE_MAP, "default": SCALE}.items()
    }

    df["sbert_bucket"] = df.apply(
        lambda row: choose_bucket(
            row["text_for_analysis"],
            row["sentiment"],
            embed_text=lambda text: sbert_model.encode([text])[0],
            scale_embeddings_by_sentiment=sbert_scale_embeddings,
        ),
        axis=1,
    )
    df["w2v_bucket"] = df.apply(
        lambda row: choose_bucket(
            row["text_for_analysis"],
            row["sentiment"],
            embed_text=lambda text: mean_embedding(w2v_model, text, 300),
            scale_embeddings_by_sentiment=w2v_scale_embeddings,
        ),
        axis=1,
    )
    df["glove_bucket"] = df.apply(
        lambda row: choose_bucket(
            row["text_for_analysis"],
            row["sentiment"],
            embed_text=lambda text: mean_embedding(glove_model, text, 300),
            scale_embeddings_by_sentiment=glove_scale_embeddings,
        ),
        axis=1,
    )

    for column in ["sbert_bucket", "w2v_bucket", "glove_bucket"]:
        df[f"{column}_label"] = df[column]
        df[column] = df[column].map(LABEL_TO_SCORE)

    df["avg_score"] = df[["sbert_bucket", "w2v_bucket", "glove_bucket"]].mean(axis=1)
    df["word_count"] = df["sentence"].astype(str).str.split().str.len()

    aspect_summary = summarize_by_aspect(df)
    review_summary = summarize_by_review(df)

    paths.scored_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.scored_output, index=False)
    aspect_summary.to_csv(paths.aspect_summary_output, index=False)
    review_summary.to_csv(paths.review_summary_output, index=False)

    return {
        "scored": df,
        "aspect_summary": aspect_summary,
        "review_summary": review_summary,
    }
