"""
Module 4: Clustering
Applies K-Means to all three representations and evaluates quality.
"""

import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AgglomerativeClustering


def _run_kmeans(X: np.ndarray,
                k: int,
                n_init: int = 20,
                random_state: int = 42) -> np.ndarray:
    """Fit K-Means++ and return cluster labels."""
    if hasattr(X, "toarray"):          # sparse → dense
        X = X.toarray()
    X = normalize(X, norm="l2")        # unit-normalise for cosine-like clustering
    km = KMeans(n_clusters=k, init="k-means++",
                n_init=n_init, random_state=random_state, max_iter=500)
    km.fit(X)
    return km.labels_, km


def cluster_stability(X: np.ndarray,
                      k: int,
                      n_runs: int = 5) -> float:
    """
    Estimate cluster stability via pairwise ARI across multiple runs.
    Returns mean ARI (1.0 = perfectly stable).
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = normalize(X, norm="l2")
    # Reduce runs at scale to control memory + time
    if X.shape[0] >= 3000:
        n_runs = 3
    labels_list = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=k, init="k-means++",
                    n_init=5 if X.shape[0] >= 3000 else 10,
                    random_state=seed, max_iter=300)
        labels_list.append(km.fit_predict(X))
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return float(np.mean(aris))

def run_hierarchical(X, k=6):
    """Run Agglomerative (hierarchical) clustering."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = normalize(X, norm="l2")

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    return labels, model


def evaluate_clustering(X: np.ndarray,
                         labels: np.ndarray,
                         true_labels=None) -> dict:
    """Compute internal (and optionally external) metrics."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = normalize(X, norm="l2")
    n_clusters = len(set(labels))
    metrics = {}
    if n_clusters > 1:
        metrics["silhouette"]    = silhouette_score(X, labels, metric="cosine",
                                                     sample_size=min(1000, len(labels)))
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    else:
        metrics["silhouette"] = metrics["davies_bouldin"] = float("nan")

    if true_labels is not None:
        metrics["ari"] = adjusted_rand_score(true_labels, labels)
        metrics["nmi"] = normalized_mutual_info_score(true_labels, labels)

    return metrics


def find_optimal_k(X: np.ndarray,
                   k_range: range = range(2, 10)) -> dict:
    """Grid-search over k; return silhouette and DB scores per k."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = normalize(X, norm="l2")
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++",
                    n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="cosine",
                               sample_size=min(500, len(labels)))
        db  = davies_bouldin_score(X, labels)
        results[k] = {"silhouette": sil, "davies_bouldin": db}
        print(f"  k={k}  silhouette={sil:.4f}  DB={db:.4f}")
    return results


def run_all_clusterings(features: dict,
                         df: pd.DataFrame,
                         k: int = 6) -> dict:
    """
    Run K-Means on tfidf_lsa, lda, and sbert representations.
    Returns a dict: name → {labels, km_model, metrics, stability}
    """
    configs = {
        "TF-IDF + KMeans":  features["tfidf_lsa"],
        "LDA + KMeans":     features["lda"],
        "SBERT + KMeans":   features["sbert"],
        "SBERT + Hierarchical": features["sbert"],   # NEW
    }

    # Optional: encode CPC codes as integer labels for external validation
    true_labels = None
    if "cpc_code" in df.columns:
        codes = df["cpc_code"].astype("category").cat.codes.values
        if len(set(codes)) > 1:
            true_labels = codes

    results = {}
    n_docs = len(df)
    for name, X in configs.items():
        # Skip Hierarchical Ward at very large scales (O(n^2) memory)
        if "Hierarchical" in name and n_docs > 6000:
            print(f"\n[Clustering] {name}  -- SKIPPED (n={n_docs} > 6000, would OOM)")
            continue
        print(f"\n[Clustering] {name}  (k={k}) ...")
        if "Hierarchical" in name:
            labels, km_model = run_hierarchical(X, k=k)
        else:
            labels, km_model = _run_kmeans(X, k=k)

        if hasattr(X, "toarray"):
            Xd = X.toarray()
        else:
            Xd = np.array(X)
        metrics = evaluate_clustering(Xd, labels, true_labels=true_labels)
        stability = cluster_stability(Xd, k=k, n_runs=5)
        metrics["stability_ari"] = stability
        print(f"  Silhouette={metrics['silhouette']:.4f}  "
              f"DB={metrics['davies_bouldin']:.4f}  "
              f"Stability={stability:.4f}")
        results[name] = {
            "labels":    labels,
            "km_model":  km_model,
            "metrics":   metrics,
        }
        del Xd
        gc.collect()
    return results


if __name__ == "__main__":
    from ingestion import load_corpus
    from preprocessing import preprocess_corpus
    from features import build_all_features
    df   = preprocess_corpus(load_corpus("synthetic"))
    feats = build_all_features(df, n_topics=6)
    res  = run_all_clusterings(feats, df, k=6)
    for name, r in res.items():
        print(f"{name}: {r['metrics']}")
