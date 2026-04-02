"""
main.py — End-to-end Patent Semantic Analysis Pipeline
Usage:
    python main.py [--source synthetic|patentsview] [--n 48] [--k 6] [--topics 6]
"""

import argparse
import time
import os

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Patent Semantic Analysis Pipeline")
    p.add_argument("--source",  default="synthetic",
                   choices=["synthetic", "patentsview"],
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
    line = "═" * (len(text) + 4)
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def main():
    args = parse_args()
    t0 = time.time()

    # ── Step 1: Ingestion ────────────────────────────────────────────────────
    banner("Step 1 · Data Ingestion")
    from ingestion import load_corpus
    df = load_corpus(source=args.source, n=args.n)
    print(f"Loaded {len(df)} patents.")

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    banner("Step 2 · Text Preprocessing")
    from preprocessing import preprocess_corpus
    df = preprocess_corpus(df)

    # ── Step 3: Feature Engineering ──────────────────────────────────────────
    banner("Step 3 · Feature Engineering")
    from features import build_all_features
    features = build_all_features(df, n_topics=args.topics)

    # ── Step 4: Clustering ───────────────────────────────────────────────────
    banner("Step 4 · Clustering")
    from clustering import run_all_clusterings
    cluster_results = run_all_clusterings(features, df, k=args.k)

    # Add cluster labels to DataFrame for each method
    for name, res in cluster_results.items():
        col = name.replace(" ", "_").replace("+", "").lower() + "_cluster"
        df[col] = res["labels"]

    # ── Step 5: Summarization ────────────────────────────────────────────────
    banner("Step 5 · Cluster Summarization")
    from summarization import build_cluster_summaries, print_cluster_report
    all_summaries = {}
    rep_map = {
        "TF-IDF + KMeans": "tfidf_lsa",
        "LDA + KMeans":    "lda",
        "SBERT + KMeans":  "sbert",
    }
    for name, res in cluster_results.items():
        X_key = rep_map[name]
        sums = build_cluster_summaries(df, res["labels"],
                                        features[X_key], name)
        all_summaries[name] = sums
        print_cluster_report(sums, name)

    # ── Step 6: Visualization ────────────────────────────────────────────────
    if not args.no_viz:
        banner("Step 6 · Visualization")
        from visualization import generate_all_visualizations
        fig_files = generate_all_visualizations(features, cluster_results, df,
                                                  k=args.k)
        print(f"\nGenerated {len(fig_files)} figures in '{OUT_DIR}/'")

    # ── Step 7: HTML Report ──────────────────────────────────────────────────
    if not args.no_report:
        banner("Step 7 · HTML Report")
        from report import generate_html_report
        report_path = generate_html_report(df, features, cluster_results,
                                            all_summaries)
        print(f"Report: {report_path}")

    # ── Save results CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "patent_clusters.csv")
    df.drop(columns=["tokens"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\nCluster assignments saved → {csv_path}")

    elapsed = time.time() - t0
    banner(f"Pipeline Complete in {elapsed:.1f}s")
    print("Outputs in:", OUT_DIR)
    print("  patent_analysis_report.html")
    print("  patent_clusters.csv")
    print("  *.png  (all figures)")


if __name__ == "__main__":
    main()
