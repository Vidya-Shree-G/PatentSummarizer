"""
main.py -- End-to-end Patent Semantic Analysis Pipeline
Usage:
    python CODE/main.py [--source synthetic|bigpatent|patentsview] [--n 48] [--k 6] [--topics 6]

Examples:
    py -3.11 CODE/main.py --source synthetic --n 48 --k 6 --topics 6
    py -3.11 CODE/main.py --source bigpatent --n 5000 --k 5 --topics 5
"""

import argparse
import time
import os
import gc
import json
import numpy as np

# Always write outputs to <repo_root>/EVALUATIONS/ regardless of CWD
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "EVALUATIONS")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Patent Semantic Analysis Pipeline")
    p.add_argument("--source",  default="synthetic",
                   choices=["synthetic", "patentsview", "bigpatent"],
                   help="Data source (default: synthetic)")
    p.add_argument("--n",       type=int, default=48,
                   help="Number of patents to load (default: 48)")
    p.add_argument("--k",       type=int, default=6,
                   help="Number of clusters (default: 6)")
    p.add_argument("--topics",  type=int, default=6,
                   help="Number of LDA topics (default: 6)")
    p.add_argument("--no-viz",  action="store_true",
                   help="Skip visualization step")
    p.add_argument("--no-report", action="store_true",
                   help="Skip HTML report generation")
    return p.parse_args()


def banner(text: str):
    line = "=" * (len(text) + 4)
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def main():
    args = parse_args()
    t0 = time.time()

    # -- Step 1: Ingestion ----------------------------------------------------
    banner("Step 1 -Data Ingestion")
    from CODE.ingestion import load_corpus
    df = load_corpus(source=args.source, n=args.n)
    print(f"Loaded {len(df)} patents.")

    # -- Step 2: Preprocessing ------------------------------------------------
    banner("Step 2 -Text Preprocessing")
    from preprocessing import preprocess_corpus
    df = preprocess_corpus(df)

    # -- Step 3: Feature Engineering ------------------------------------------
    banner("Step 3 -Feature Engineering")
    from features import build_all_features
    from features import find_best_topics
    print("\n=== Finding optimal LDA topics ===")
    best_topics, topic_scores = find_best_topics(df)
    gc.collect()

    features = build_all_features(df, n_topics=best_topics)
    gc.collect()

    # -- Step 4: Clustering ---------------------------------------------------
    banner("Step 4 -Clustering")
    from clustering import run_all_clusterings
    from clustering import find_optimal_k

    print("\n=== Finding optimal k (SBERT) ===")
    k_results = find_optimal_k(features["sbert"], range(2, 10))

    silhouette_best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
    print(f"Silhouette-optimal k (diagnostic): {silhouette_best_k}")
    print(f"Using user-specified k = {args.k} for clustering "
          f"(matches CPC domain count for interpretability)")
    best_k = args.k

    cluster_results = run_all_clusterings(features, df, k=best_k)

    print("\n=== Model Comparison ===")

    for name, res in cluster_results.items():
        m = res["metrics"]
        print(
            f"{name}: "
            f"Sil={m['silhouette']:.3f}, "
            f"DB={m['davies_bouldin']:.3f}, "
            f"Stab={m['stability_ari']:.3f}"
        )

    # Find best model
    best_model = max(
        cluster_results,
        key=lambda x: cluster_results[x]["metrics"]["silhouette"]
    )

    print(f"\nBest performing model (Silhouette): {best_model}")

    # Add cluster labels to DataFrame for each method
    for name, res in cluster_results.items():
        col = name.replace(" ", "_").replace("+", "").lower() + "_cluster"
        df[col] = res["labels"]

    # -- Step 5: Summarization ------------------------------------------------
    banner("Step 5 -Cluster Summarization")
    from summarization import build_cluster_summaries, print_cluster_report, evaluate_summaries
    all_summaries = {}
    summary_scores = {}
    rep_map = {
        "TF-IDF + KMeans": "tfidf_lsa",
        "LDA + KMeans":    "lda",
        "SBERT + KMeans":  "sbert",
        "SBERT + Hierarchical": "sbert",
    }
    for name, res in cluster_results.items():
        X_key = rep_map[name]
        sums = build_cluster_summaries(df, res["labels"],
                                        features[X_key], name)
        all_summaries[name] = sums
        print_cluster_report(sums, name)
        score = evaluate_summaries(sums, df)
        summary_scores[name] = score

        print(
            f"[Summarization] {name} | "
            f"R1={score['rouge1']:.3f} "
            f"R2={score['rouge2']:.3f} "
            f"RL={score['rougeL']:.3f} "
            f"Cov={score['coverage']:.3f} "
            f"Cent={score['proximity']:.3f}"
        )

    # -- Step 6: Visualization ------------------------------------------------
    if not args.no_viz:
        banner("Step 6 -Visualization")
        from visualization import generate_all_visualizations
        fig_files = generate_all_visualizations(features, cluster_results, df,
                                                  k=best_k)
        print(f"\nGenerated {len(fig_files)} figures in '{OUT_DIR}/'")

    # -- Step 7: HTML Report --------------------------------------------------
    if not args.no_report:
        banner("Step 7 -HTML Report")
        from report import generate_html_report
        report_path = generate_html_report(df, features, cluster_results,
                                            all_summaries, summary_scores=summary_scores)
        print(f"Report: {report_path}")

    # -- Save results CSV -----------------------------------------------------
    csv_path = os.path.join(OUT_DIR, "patent_clusters.csv")
    df.drop(columns=["tokens"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\nCluster assignments saved -> {csv_path}")

    # -- Save metrics + summaries JSON for Streamlit demo --------------------
    results_json = {
        "source":      args.source,
        "n_patents":   int(len(df)),
        "k":           int(best_k),
        "n_topics":    int(best_topics),
        "best_model":  best_model,
        "metrics":     {name: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                              for kk, vv in res["metrics"].items()
                              if not hasattr(vv, "shape")}
                       for name, res in cluster_results.items()},
        "summary_scores": summary_scores,
        "summaries":   {name: [{kk: (vv if not hasattr(vv, "tolist") else vv.tolist())
                               for kk, vv in s.items()} for s in sums]
                       for name, sums in all_summaries.items()},
        "cpc_counts":  df["cpc_code"].value_counts().to_dict() if "cpc_code" in df.columns else {},
        "lda_coherence": float(features.get("lda_coherence", 0.0)),
        "elapsed_sec": float(time.time() - t0),
    }
    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"Metrics JSON saved -> {json_path}")

    elapsed = time.time() - t0
    banner(f"Pipeline Complete in {elapsed:.1f}s")
    print("Outputs in:", OUT_DIR)
    print("  patent_analysis_report.html")
    print("  patent_clusters.csv")
    print("  *.png  (all figures)")


if __name__ == "__main__":
    main()
