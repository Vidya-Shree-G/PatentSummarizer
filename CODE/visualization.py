"""
Module 6: Visualization -- t-SNE, PCA, metrics, LDA topic bars, heatmap
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "EVALUATIONS")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261",
           "#264653","#A8DADC","#6D6875","#B5E48C","#023E8A"]

def _colors(labels):
    u = sorted(set(labels))
    cm = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(u)}
    return [cm[l] for l in labels], cm

def _reduce(X, method="tsne", n=5):
    if hasattr(X, "toarray"): X = X.toarray()
    X = normalize(np.array(X, dtype=float), norm="l2")
    nc = min(30, X.shape[1]-1, X.shape[0]-1)
    if nc > 2:
        X = PCA(n_components=nc, random_state=42).fit_transform(X)
    if method == "tsne":
        perp = min(n, max(2, X.shape[0]//4))
        return TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000).fit_transform(X)
    return PCA(n_components=2, random_state=42).fit_transform(X)

def plot_clusters(X, labels, titles, name, proj="tsne", filename=None):
    print(f"[Viz] {name} -- {proj.upper()} ...")
    X2d = _reduce(X, method=proj)
    cols, cmap = _colors(labels)
    fig, ax = plt.subplots(figsize=(9,7))
    ax.scatter(X2d[:,0], X2d[:,1], c=cols, s=60, alpha=0.85,
               linewidths=0.3, edgecolors="white")
    for i in range(min(8, len(X2d))):
        ax.annotate(str(titles[i])[:28], (X2d[i,0], X2d[i,1]),
                    fontsize=6, alpha=0.7)
    patches = [mpatches.Patch(color=cmap[c], label=f"Cluster {c}")
               for c in sorted(cmap)]
    ax.legend(handles=patches, loc="best", fontsize=8)
    ax.set_title(f"{name} -- {proj.upper()}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fn = filename or f"{OUT_DIR}/{name.replace(' ','_')}_{proj}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {fn}")
    return fn

def plot_all_projections(features, cluster_results, df, proj="tsne"):
    configs = [("TF-IDF + KMeans","tfidf_lsa"),
               ("LDA + KMeans","lda"),
               ("SBERT + KMeans","sbert"),
               ("SBERT + Hierarchical","sbert"),]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    titles = df["title"].tolist()
    for ax, (name, key) in zip(axes, configs):
        labels = cluster_results[name]["labels"]
        X2d = _reduce(features[key], method=proj)
        cols, cmap = _colors(labels)
        ax.scatter(X2d[:,0], X2d[:,1], c=cols, s=50, alpha=0.8,
                   linewidths=0.3, edgecolors="white")
        patches = [mpatches.Patch(color=cmap[c], label=f"C{c}") for c in sorted(cmap)]
        ax.legend(handles=patches, fontsize=7, loc="best")
        ax.set_title(name, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle(f"Cluster Projections ({proj.upper()})", fontsize=14, y=1.01)
    plt.tight_layout()
    fn = f"{OUT_DIR}/all_projections_{proj}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Viz] -> {fn}"); return fn

def plot_metrics_comparison(cluster_results):
    methods = list(cluster_results.keys())
    sil  = [cluster_results[m]["metrics"].get("silhouette", 0) for m in methods]
    db   = [cluster_results[m]["metrics"].get("davies_bouldin", 0) for m in methods]
    stab = [cluster_results[m]["metrics"].get("stability_ari", 0) for m in methods]
    x, w = np.arange(len(methods)), 0.25
    fig, ax = plt.subplots(figsize=(10,5))
    b1 = ax.bar(x-w, sil,  w, label="Silhouette ↑",  color="#2A9D8F", alpha=0.85)
    b2 = ax.bar(x,   db,   w, label="Davies-Bouldin ↓", color="#E63946", alpha=0.85)
    b3 = ax.bar(x+w, stab, w, label="Stability ARI ↑",  color="#457B9D", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=10, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Clustering Metrics", fontsize=13)
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (b1,b2,b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0,3), textcoords="offset points", ha="center", fontsize=7)
    plt.tight_layout()
    fn = f"{OUT_DIR}/metrics_comparison.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Viz] -> {fn}"); return fn

def plot_lda_topics(lda_model, n_topics=6, n_words=8):
    vocab = lda_model._vocab
    cols = int(np.ceil(np.sqrt(n_topics)))
    rows = int(np.ceil(n_topics / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(n_topics):
        ax = axes_flat[i]
        comp = lda_model.components_[i]
        top  = np.argsort(comp)[::-1][:n_words]
        words  = vocab[top][::-1]
        scores = comp[top][::-1]
        ax.barh(words, scores, color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax.set_title(f"Topic {i}", fontsize=10)
        ax.tick_params(labelsize=8)
    for j in range(n_topics, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("LDA Topics -- Top Words", fontsize=13)
    plt.tight_layout()
    fn = f"{OUT_DIR}/lda_topics.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Viz] -> {fn}"); return fn

def plot_cluster_sizes(cluster_results):
    n = len(cluster_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, (name, res) in zip(axes, cluster_results.items()):
        labels = res["labels"]
        unique, counts = np.unique(labels, return_counts=True)
        ax.bar([f"C{c}" for c in unique], counts,
               color=[PALETTE[i%len(PALETTE)] for i in range(len(unique))],
               alpha=0.85, edgecolor="white")
        ax.set_title(name, fontsize=10); ax.set_ylabel("# Patents")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Cluster Size Distribution", fontsize=13)
    plt.tight_layout()
    fn = f"{OUT_DIR}/cluster_sizes.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Viz] -> {fn}"); return fn

def plot_optimal_k(k_results, method_name="SBERT"):
    ks  = sorted(k_results.keys())
    sil = [k_results[k]["silhouette"] for k in ks]
    db  = [k_results[k]["davies_bouldin"] for k in ks]
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.plot(ks, sil, "o-", color="#2A9D8F", label="Silhouette ↑")
    ax2.plot(ks, db,  "s--", color="#E63946", label="Davies-Bouldin ↓")
    ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette", color="#2A9D8F")
    ax2.set_ylabel("Davies-Bouldin", color="#E63946")
    ax1.set_title(f"Optimal k -- {method_name}", fontsize=12)
    lines1,labs1 = ax1.get_legend_handles_labels()
    lines2,labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fn = f"{OUT_DIR}/optimal_k_{method_name.replace(' ','_')}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Viz] -> {fn}"); return fn

def plot_doc_topic_heatmap(lda_mat, df, max_docs=30):
    n = min(max_docs, len(df))
    data = lda_mat[:n]
    short = [str(t)[:22] for t in df["title"].tolist()[:n]]
    fig, ax = plt.subplots(figsize=(10, max(4, n*0.3)))
    sns.heatmap(data, ax=ax, cmap="YlOrRd", linewidths=0.3,
                xticklabels=[f"T{i}" for i in range(data.shape[1])],
                yticklabels=short, vmin=0, vmax=1)
    ax.set_title("Document-Topic Heatmap (LDA)", fontsize=12)
    plt.tight_layout()
    fn = f"{OUT_DIR}/doc_topic_heatmap.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Viz] -> {fn}"); return fn

def generate_all_visualizations(features, cluster_results, df, k=6):
    files = []
    titles = df["title"].tolist()
    for name, key in [("TF-IDF + KMeans","tfidf_lsa"),
                       ("LDA + KMeans","lda"),
                       ("SBERT + KMeans","sbert"),
                       ("SBERT + Hierarchical","sbert"),]:
        labels = cluster_results[name]["labels"]
        X = features[key]
        files.append(plot_clusters(X, labels, titles, name, proj="tsne"))
        files.append(plot_clusters(X, labels, titles, name, proj="pca",
                     filename=f"{OUT_DIR}/{name.replace(' ','_')}_pca.png"))
    files.append(plot_all_projections(features, cluster_results, df, proj="tsne"))
    files.append(plot_all_projections(features, cluster_results, df, proj="pca"))
    files.append(plot_metrics_comparison(cluster_results))
    files.append(plot_cluster_sizes(cluster_results))
    lda_k = features["lda"].shape[1]
    files.append(plot_lda_topics(features["lda_model"], n_topics=lda_k))
    files.append(plot_doc_topic_heatmap(features["lda"], df))

    from clustering import find_optimal_k
    print("\n[Viz] Optimal k search on SBERT ...")
    k_res = find_optimal_k(features["sbert"], k_range=range(2, min(10, len(df))))
    files.append(plot_optimal_k(k_res, "SBERT"))
    return files
