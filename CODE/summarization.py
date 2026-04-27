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
from collections import Counter


# -- Sentence-level extractive summarizer ------------------------------------

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


# -- Top keywords per cluster via TF-IDF -------------------------------------

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


# -- Representative document selection ---------------------------------------

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


# -- Cluster report builder ---------------------------------------------------

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


# -- Rogue 1 ---------------------------------------------------
def rouge1_score(summary, references):
    """
    Simple ROUGE-1 recall: overlap of unigrams
    """
    summary_tokens = summary.lower().split()
    ref_tokens = " ".join(references).lower().split()

    summary_counts = Counter(summary_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = sum(min(summary_counts[w], ref_counts[w]) for w in summary_counts)

    return overlap / max(1, len(ref_tokens))

# -- Rouge 2 ---------------------------------------------------
def _bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

def rouge2_score(summary, references):
    summary_tokens = summary.lower().split()
    ref_tokens = " ".join(references).lower().split()

    summary_bi = Counter(_bigrams(summary_tokens))
    ref_bi = Counter(_bigrams(ref_tokens))

    overlap = sum(min(summary_bi[k], ref_bi[k]) for k in summary_bi)
    return overlap / max(1, sum(ref_bi.values()))

def _lcs(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]

# -- Rougue L ---------------------------------------------------
def rouge_l_score(summary, references):
    summary_tokens = summary.lower().split()
    ref_tokens = " ".join(references).lower().split()
    lcs_len = _lcs(summary_tokens, ref_tokens)
    return lcs_len / max(1, len(ref_tokens))


def keyword_coverage(summary, keywords):
    summary_words = set(summary.lower().split())
    keywords = set([k.lower() for k in keywords])
    if not keywords:
        return 0.0
    return len(summary_words & keywords) / len(keywords)


def centroid_proximity(summary, cluster_texts):
    if not cluster_texts:
        return 0.0

    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    texts = cluster_texts + [summary]
    tfidf = vec.fit_transform(texts)

    # Force proper numpy arrays (no matrix possible)
    cluster_vec = tfidf[:-1].toarray()
    cluster_vec = np.mean(cluster_vec, axis=0).reshape(1, -1)  

    summary_vec = tfidf[-1].toarray().reshape(1, -1)           

    sim = cosine_similarity(summary_vec, cluster_vec)[0][0]
    return float(sim)

def evaluate_summaries(summaries, df):
    r1, r2, rl = [], [], []
    cov, prox = [], []

    for s in summaries:
        refs = s["representative_titles"]
        r1.append(rouge1_score(s["summary"], refs))
        r2.append(rouge2_score(s["summary"], refs))
        rl.append(rouge_l_score(s["summary"], refs))

        cov.append(keyword_coverage(s["summary"], s["keywords"]))

        # cluster texts for proximity
        cluster_texts = df[df["title"].isin(refs)]["text"].tolist()

        if not cluster_texts:
            prox.append(0.0)
        else:
            prox.append(centroid_proximity(s["summary"], cluster_texts))

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
        "coverage": float(np.mean(cov)),
        "proximity": float(np.mean(prox))
    }


def print_cluster_report(summaries: list[dict], method_name: str = ""):
    header = f"{'-'*60}\n CLUSTER REPORT -- {method_name}\n{'-'*60}"
    print(header)
    for s in summaries:
        print(f"\nCluster {s['cluster_id']}  ({s['size']} patents)")
        print(f"  Keywords : {', '.join(s['keywords'][:6])}")
        print(f"  Summary  : {s['summary'][:250]} ...")
        if s["representative_titles"]:
            for t in s["representative_titles"]:
                print(f"  * {t}")
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
        score = evaluate_summaries(sums,df)
        print(f"[Summarization] {name} ROUGE-1: {score:.4f}\n")
