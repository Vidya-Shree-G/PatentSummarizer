"""
Module 5: Summarization
Generates cluster-level summaries using extractive TF-IDF sentence ranking
and centroid-proximity representative selection.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Sentence-level extractive summarizer ────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristic."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 5]


def extractive_summarize(texts: list[str],
                          n_sentences: int = 3) -> str:
    """
    Given a list of documents, return an extractive summary
    by ranking sentences via TF-IDF cosine similarity to centroid.
    """
    combined = " ".join(texts)
    sentences = _split_sentences(combined)
    if not sentences:
        return combined[:300]
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        tfidf = vec.fit_transform(sentences)
        centroid = tfidf.mean(axis=0)
        scores = cosine_similarity(tfidf, centroid).flatten()
        top_idx = np.argsort(scores)[::-1][:n_sentences]
        top_idx_sorted = sorted(top_idx)
        return " ".join(sentences[i] for i in top_idx_sorted)
    except Exception:
        return combined[:400]


# ── Top keywords per cluster via TF-IDF ─────────────────────────────────────

def cluster_top_keywords(cluster_texts: list[str],
                          corpus_texts: list[str],
                          n_keywords: int = 8) -> list[str]:
    """
    Return the top TF-IDF keywords that characterise a cluster
    relative to the full corpus.
    """
    try:
        vec = TfidfVectorizer(stop_words="english",
                               max_features=3000, ngram_range=(1, 2))
        vec.fit(corpus_texts)
        cluster_doc = " ".join(cluster_texts)
        tfidf = vec.transform([cluster_doc])
        scores = np.asarray(tfidf.todense()).flatten()
        top_idx = np.argsort(scores)[::-1][:n_keywords]
        terms = np.array(vec.get_feature_names_out())
        return list(terms[top_idx])
    except Exception:
        return []


# ── Representative document selection ───────────────────────────────────────

def find_representatives(X: np.ndarray,
                          labels: np.ndarray,
                          df: pd.DataFrame,
                          n_per_cluster: int = 2) -> dict[int, list[int]]:
    """
    For each cluster, find the n documents closest to the centroid.
    Returns {cluster_id: [row_indices]}.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=float)
    reps = {}
    for cid in np.unique(labels):
        mask = labels == cid
        cluster_X = X[mask]
        centroid   = cluster_X.mean(axis=0, keepdims=True)
        sims       = cosine_similarity(cluster_X, centroid).flatten()
        top_local  = np.argsort(sims)[::-1][:n_per_cluster]
        global_idx = np.where(mask)[0][top_local]
        reps[int(cid)] = list(global_idx)
    return reps


# ── Cluster report builder ───────────────────────────────────────────────────

def build_cluster_summaries(df: pd.DataFrame,
                              labels: np.ndarray,
                              X: np.ndarray,
                              method_name: str = "Model") -> list[dict]:
    """
    Build a human-readable summary for every cluster.
    Returns a list of dicts with keys:
      cluster_id, size, keywords, summary, representative_titles
    """
    corpus_texts = df["text"].tolist()
    reps_map     = find_representatives(X, labels, df)
    summaries    = []

    for cid in sorted(np.unique(labels)):
        mask         = labels == cid
        cluster_df   = df[mask]
        cluster_texts = cluster_df["text"].tolist()
        keywords     = cluster_top_keywords(cluster_texts, corpus_texts)
        summary      = extractive_summarize(cluster_texts, n_sentences=2)
        rep_idxs     = reps_map.get(cid, [])
        rep_titles   = df.iloc[rep_idxs]["title"].tolist() if rep_idxs else []

        summaries.append({
            "cluster_id":           int(cid),
            "size":                 int(mask.sum()),
            "keywords":             keywords,
            "summary":              summary,
            "representative_titles": rep_titles,
        })

    print(f"[Summarization] Built {len(summaries)} cluster summaries for {method_name}.")
    return summaries


def print_cluster_report(summaries: list[dict], method_name: str = ""):
    header = f"{'─'*60}\n CLUSTER REPORT — {method_name}\n{'─'*60}"
    print(header)
    for s in summaries:
        print(f"\nCluster {s['cluster_id']}  ({s['size']} patents)")
        print(f"  Keywords : {', '.join(s['keywords'][:6])}")
        print(f"  Summary  : {s['summary'][:250]} …")
        if s["representative_titles"]:
            for t in s["representative_titles"]:
                print(f"  ★ {t}")
    print()


if __name__ == "__main__":
    from ingestion import load_corpus
    from preprocessing import preprocess_corpus
    from features import build_all_features
    from clustering import run_all_clusterings
    df    = preprocess_corpus(load_corpus("synthetic"))
    feats = build_all_features(df, n_topics=6)
    clust = run_all_clusterings(feats, df, k=6)
    for name, res in clust.items():
        sums = build_cluster_summaries(df, res["labels"],
                                        feats["sbert"], name)
        print_cluster_report(sums, name)
