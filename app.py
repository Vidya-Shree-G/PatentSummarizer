"""
Streamlit demo app for the Patent Corpus Semantic Analysis pipeline.
Run with:  streamlit run app.py
"""

import os
import sys
import json
import time
import io
import contextlib
import pandas as pd
import streamlit as st
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────────

ROOT     = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "CODE")
EVAL_DIR = os.path.join(ROOT, "EVALUATIONS")
DATA_DIR = os.path.join(ROOT, "DATA")
sys.path.insert(0, CODE_DIR)


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Patent Corpus Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
.main-title  { color:#0f3460; font-size:2.4rem; font-weight:800; margin-bottom:0; }
.subtitle    { color:#16213e; font-size:1.05rem; margin-top:.1rem; opacity:.85; }
.attribution { color:#666; font-size:.85rem; margin-bottom:1rem; }
.stat-card {
    background:linear-gradient(135deg,#0f3460,#16213e);
    color:white;
    padding:14px 18px;
    border-radius:10px;
    text-align:center;
    box-shadow:0 2px 8px rgba(15,52,96,.15);
}
.stat-val { font-size:1.6rem; font-weight:700; margin-bottom:2px; }
.stat-lbl { font-size:.78rem; opacity:.8; text-transform:uppercase; letter-spacing:.5px; }
.cluster-card {
    background:white;
    border:1px solid #e0e6f0;
    border-left:4px solid #0f3460;
    border-radius:8px;
    padding:14px 18px;
    margin:8px 0;
    box-shadow:0 1px 4px rgba(0,0,0,.04);
}
.cluster-card h4 { color:#0f3460; margin:0 0 8px 0; font-size:1.05rem; }
.kw-chip {
    display:inline-block;
    background:#e8f4f8;
    color:#0f3460;
    border-radius:14px;
    padding:3px 11px;
    font-size:.82rem;
    margin:2px 3px 2px 0;
    font-weight:500;
}
.rep-title {
    background:#fff8e1;
    border-left:3px solid #e9c46a;
    padding:6px 12px;
    margin:4px 0;
    font-size:.86rem;
    border-radius:0 4px 4px 0;
    color:#333;
}
.summary-text { color:#555; font-style:italic; font-size:.9rem; margin:6px 0; }
.best-badge   {
    background:#d4edda; color:#155724;
    padding:2px 10px; border-radius:12px;
    font-size:.75rem; font-weight:700;
}
hr.section { border:none; border-top:2px solid #e0e6f0; margin:20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────────

c1, c2 = st.columns([5, 1])
with c1:
    st.markdown('<div class="main-title">Patent Corpus Semantic Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">TF-IDF · LDA · Sentence-BERT · K-Means · '
                'Hierarchical Clustering · t-SNE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="attribution">Group 22 · Project 9 · CSE 573 Spring 2026 · '
                'Arizona State University</div>',
                unsafe_allow_html=True)
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "[GitHub Repo](https://github.com/Vidya-Shree-G/PatentSummarizer)",
        unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_results_json():
    path = os.path.join(EVAL_DIR, "results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_clusters_csv():
    path = os.path.join(EVAL_DIR, "patent_clusters.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def show_image(name, caption=None, use_container_width=True):
    path = os.path.join(EVAL_DIR, name)
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=use_container_width)
    else:
        st.warning(f"Image not found: {name}")

def stat_card(label, value):
    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
    return f'<div class="stat-card"><div class="stat-val">{val_str}</div>' \
           f'<div class="stat-lbl">{label}</div></div>'


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Demo Controls")

    mode = st.radio(
        "Display mode",
        ["📊 View saved results", "▶ Run pipeline live"],
        help="Saved = pre-computed results from EVALUATIONS/. Live = run the pipeline now."
    )

    st.markdown("---")
    if mode == "▶ Run pipeline live":
        st.markdown("**Pipeline Parameters**")
        source = st.selectbox(
            "Data source",
            ["synthetic", "bigpatent"],
            help="synthetic = up to 48 patents (~10s).  "
                 "bigpatent = real USPTO patents (500=~30s, 5000=~6min)."
        )
        if source == "synthetic":
            n_patents = st.slider("Patents to load", 24, 48, 48, step=12)
        else:
            n_patents = st.slider("Patents to load", 500, 5000, 500, step=500,
                                  help="500 demos in ~30s. 5000 takes ~6 min "
                                       "(needs 1.5GB+ free RAM).")
        k_clusters = st.slider("Clusters (k)", 2, 10, 5)
        n_topics = st.slider("LDA topics", 2, 10, 5)
        skip_viz = st.checkbox("Skip charts (faster)", value=False)
        run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)
    else:
        run_btn = False

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "End-to-end pipeline that ingests, clusters, and summarises patent "
        "corpora using three feature representations (TF-IDF, LDA, "
        "Sentence-BERT) and two clustering algorithms (K-Means++, "
        "Hierarchical-Ward)."
    )


# ── Run pipeline live ───────────────────────────────────────────────────────

if mode == "▶ Run pipeline live" and run_btn:
    progress = st.progress(0, text="Starting pipeline...")
    log_box = st.expander("📜 Live pipeline log", expanded=False)
    output_buffer = io.StringIO()

    # Capture stdout
    with contextlib.redirect_stdout(output_buffer):
        try:
            t0 = time.time()
            from ingestion import load_corpus
            progress.progress(10, text="Step 1/7 — Loading corpus...")
            df_live = load_corpus(source=source, n=n_patents)

            from preprocessing import preprocess_corpus
            progress.progress(20, text="Step 2/7 — Preprocessing...")
            df_live = preprocess_corpus(df_live)

            from features import build_all_features
            progress.progress(35, text="Step 3/7 — Building features (TF-IDF + LDA + SBERT)...")
            features = build_all_features(df_live, n_topics=n_topics)

            from clustering import run_all_clusterings
            progress.progress(60, text="Step 4/7 — Clustering...")
            cluster_results = run_all_clusterings(features, df_live, k=k_clusters)

            from summarization import build_cluster_summaries, evaluate_summaries
            progress.progress(75, text="Step 5/7 — Generating summaries...")

            rep_map = {
                "TF-IDF + KMeans":      "tfidf_lsa",
                "LDA + KMeans":         "lda",
                "SBERT + KMeans":       "sbert",
                "SBERT + Hierarchical": "sbert",
            }
            all_summaries  = {}
            summary_scores = {}
            for name, res in cluster_results.items():
                X_key = rep_map[name]
                sums = build_cluster_summaries(df_live, res["labels"],
                                                features[X_key], name)
                all_summaries[name]  = sums
                summary_scores[name] = evaluate_summaries(sums, df_live)

            progress.progress(95, text="Step 6/7 — Finalizing...")
            elapsed = time.time() - t0
            progress.progress(100, text=f"✓ Pipeline complete in {elapsed:.1f}s")

            # Save to session state
            st.session_state["live_results"] = {
                "source":         source,
                "n_patents":      len(df_live),
                "k":              k_clusters,
                "n_topics":       n_topics,
                "metrics":        {n: r["metrics"] for n, r in cluster_results.items()},
                "summary_scores": summary_scores,
                "summaries":      all_summaries,
                "df":             df_live,
                "elapsed":        elapsed,
            }
            st.success(f"✅ Pipeline ran in {elapsed:.1f}s — {len(df_live)} patents, "
                       f"k={k_clusters}, {n_topics} LDA topics")
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())

    with log_box:
        st.code(output_buffer.getvalue() or "(no output)")


# ── Pick data source for display ────────────────────────────────────────────

if "live_results" in st.session_state and mode == "▶ Run pipeline live":
    R = st.session_state["live_results"]
    is_live = True
else:
    R = load_results_json()
    is_live = False
    if R is None:
        st.error("⚠ No saved results found in EVALUATIONS/results.json. "
                 "Run the pipeline first: `py -3.11 CODE/main.py --source bigpatent`")
        st.stop()


# ── Top stat cards ──────────────────────────────────────────────────────────

st.markdown(f'<hr class="section">', unsafe_allow_html=True)

cards = st.columns(5)
metrics_dict = R["metrics"]

# Best models
best_clust = max(metrics_dict, key=lambda x: metrics_dict[x].get("silhouette", -1))
best_sum   = max(R["summary_scores"], key=lambda x: R["summary_scores"][x].get("rouge1", -1))

with cards[0]:
    st.markdown(stat_card("Patents", R["n_patents"]), unsafe_allow_html=True)
with cards[1]:
    st.markdown(stat_card("Clusters", R["k"]), unsafe_allow_html=True)
with cards[2]:
    st.markdown(stat_card("LDA Topics", R["n_topics"]), unsafe_allow_html=True)
with cards[3]:
    st.markdown(stat_card("Best Silhouette",
                          metrics_dict[best_clust]["silhouette"]),
                unsafe_allow_html=True)
with cards[4]:
    st.markdown(stat_card("Best ROUGE-1",
                          R["summary_scores"][best_sum]["rouge1"]),
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Corpus",
    "🧪 Features & Topics",
    "🎯 Clustering",
    "📝 Cluster Summaries",
    "🗺 Visualizations",
])


# ── Tab 1: Corpus Overview ──────────────────────────────────────────────────

with tab1:
    st.subheader("Corpus Overview")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Source:** `{R.get('source', 'unknown')}`")
        st.markdown(f"**Total patents:** {R['n_patents']}")
        st.markdown(f"**Best clustering model:** {best_clust} "
                    f"(Silhouette = {metrics_dict[best_clust]['silhouette']:.3f})")
        st.markdown(f"**Best summarization model:** {best_sum} "
                    f"(ROUGE-1 = {R['summary_scores'][best_sum]['rouge1']:.3f})")

    with col2:
        cpc = R.get("cpc_counts", {})
        if cpc:
            st.markdown("**CPC Class Distribution**")
            cpc_df = pd.DataFrame([
                {"CPC Code": k, "Patent Count": v} for k, v in cpc.items()
            ])
            st.dataframe(cpc_df, hide_index=True, use_container_width=True)

    if not is_live:
        df_csv = load_clusters_csv()
        if df_csv is not None:
            st.markdown("**Sample patents** (first 8 rows)")
            cols_show = [c for c in ["patent_id", "title", "cpc_code"] if c in df_csv.columns]
            st.dataframe(df_csv[cols_show].head(8), hide_index=True,
                         use_container_width=True)


# ── Tab 2: Features ─────────────────────────────────────────────────────────

with tab2:
    st.subheader("Feature Engineering")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### TF-IDF + LSA")
        st.markdown(
            "- Sparse lexical vectors\n"
            "- Top 3,000 terms (1- and 2-grams)\n"
            "- SVD reduction → 50 dims\n"
            "- Fast, interpretable"
        )
    with col2:
        st.markdown("##### LDA Topic Model")
        st.markdown(
            f"- {R['n_topics']} latent topics\n"
            f"- Coherence = {R.get('lda_coherence', 0):.4f}\n"
            "- Document-topic distribution\n"
            "- Discovers semantic themes"
        )
    with col3:
        st.markdown("##### Sentence-BERT")
        st.markdown(
            "- `all-MiniLM-L6-v2`\n"
            "- Dense 384-dim embeddings\n"
            "- Context-aware\n"
            "- Captures paraphrases"
        )

    if not is_live:
        st.markdown("---")
        st.markdown("##### LDA Topic Word Distributions")
        show_image("lda_topics.png")

        st.markdown("##### Document × Topic Probability Heatmap")
        show_image("doc_topic_heatmap.png")


# ── Tab 3: Clustering ───────────────────────────────────────────────────────

with tab3:
    st.subheader("Clustering Evaluation")
    st.caption("Silhouette ↑ better · Davies-Bouldin ↓ better · Stability ARI ↑ better")

    rows = []
    for method, m in metrics_dict.items():
        rows.append({
            "Method":              method,
            "Silhouette ↑":        round(m.get("silhouette", 0), 4),
            "Davies-Bouldin ↓":    round(m.get("davies_bouldin", 0), 4),
            "Stability ARI ↑":     round(m.get("stability_ari", 0), 4),
            "Best": "🏆" if method == best_clust else "",
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.success(f"🏆 **Best model:** {best_clust} — Silhouette "
               f"{metrics_dict[best_clust]['silhouette']:.4f}, "
               f"Davies-Bouldin {metrics_dict[best_clust]['davies_bouldin']:.4f}, "
               f"Stability {metrics_dict[best_clust]['stability_ari']:.4f}")

    if not is_live:
        st.markdown("---")
        st.markdown("##### Metrics Comparison")
        show_image("metrics_comparison.png")

        st.markdown("##### Cluster Size Distribution")
        show_image("cluster_sizes.png")

        st.markdown("##### Optimal k Search (SBERT)")
        show_image("optimal_k_SBERT.png")


# ── Tab 4: Cluster Summaries ────────────────────────────────────────────────

with tab4:
    st.subheader("Cluster Summaries & ROUGE Evaluation")

    rows = []
    for method, s in R["summary_scores"].items():
        rows.append({
            "Method":          method,
            "ROUGE-1":         round(s.get("rouge1", 0), 4),
            "ROUGE-2":         round(s.get("rouge2", 0), 4),
            "ROUGE-L":         round(s.get("rougeL", 0), 4),
            "Kw Coverage":     round(s.get("coverage", 0), 4),
            "Centroid Sim":    round(s.get("proximity", 0), 4),
            "Best": "🏆" if method == best_sum else "",
        })
    sum_df = pd.DataFrame(rows)
    st.dataframe(sum_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    method_names = list(R["summaries"].keys())
    chosen_method = st.selectbox("Show summaries for:", method_names,
                                  index=method_names.index(best_clust)
                                  if best_clust in method_names else 0)

    summaries = R["summaries"][chosen_method]
    for s in summaries:
        chips = "".join(f'<span class="kw-chip">{k}</span>'
                        for k in s.get("keywords", [])[:8])
        reps = "".join(f'<div class="rep-title">★ {t}</div>'
                       for t in s.get("representative_titles", []))
        st.markdown(f"""
        <div class="cluster-card">
          <h4>Cluster {s['cluster_id']} — {s['size']} patents</h4>
          <div style="margin-bottom:8px">{chips}</div>
          <p class="summary-text">{(s.get('summary') or '')[:400]}…</p>
          {reps}
        </div>
        """, unsafe_allow_html=True)


# ── Tab 5: Visualizations ───────────────────────────────────────────────────

with tab5:
    st.subheader("Dimensionality Reduction Visualizations")

    if is_live:
        st.info("📊 t-SNE / PCA visualisations are available only in **View saved results** "
                "mode. Run `py -3.11 CODE/main.py` (without `--no-viz`) to generate them.")
    else:
        st.markdown("##### All Methods — t-SNE Side-by-Side")
        show_image("all_projections_tsne.png")

        st.markdown("##### All Methods — PCA Side-by-Side")
        show_image("all_projections_pca.png")

        st.markdown("---")
        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown("**TF-IDF + KMeans (t-SNE)**")
            show_image("TF-IDF_+_KMeans_tsne.png")
            st.markdown("**LDA + KMeans (t-SNE)**")
            show_image("LDA_+_KMeans_tsne.png")
        with sub2:
            st.markdown("**SBERT + KMeans (t-SNE)**")
            show_image("SBERT_+_KMeans_tsne.png")
            st.markdown("**SBERT + Hierarchical (t-SNE)**")
            show_image("SBERT_+_Hierarchical_tsne.png")


# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown('<hr class="section">', unsafe_allow_html=True)
st.caption(
    "Pipeline source: `CODE/`  ·  Outputs: `EVALUATIONS/`  ·  "
    "Group 22 · Project 9 · CSE 573 Spring 2026 · Arizona State University"
)
