"""
Module 7: Report Generator
Self-contained HTML report with working sticky nav + smooth scroll offset.
"""

import os, base64, json
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "EVALUATIONS")


def _img_tag(path, alt="", width="100%"):
    if not os.path.exists(path):
        return f'<p style="color:#aaa;font-style:italic;">⚠ Image not found: {os.path.basename(path)}</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
            f'style="width:{width};border-radius:8px;'
            f'box-shadow:0 2px 10px rgba(0,0,0,.15);margin:10px 0;">')


def _metric_card(label, value):
    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
    return (f'<div class="metric-card">'
            f'<div class="metric-val">{val_str}</div>'
            f'<div class="metric-lbl">{label}</div>'
            f'</div>')


def generate_html_report(df, features, cluster_results, all_summaries,
                          output_path=None,summary_scores=None):
    output_path = output_path or os.path.join(OUT_DIR, "patent_analysis_report.html")

    NAV_H = 52   # px -- must match nav height in CSS

    css = f"""
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9;
        color:#222; line-height:1.6; }}

/* -- Header -- */
header {{ background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
          color:#fff; padding:40px 48px; }}
header h1 {{ font-size:2rem; margin-bottom:6px; }}
header p  {{ opacity:.8; font-size:.95rem; }}

/* -- Sticky nav -- */
nav {{
  background:#0f3460;
  padding:0 48px;
  display:flex;
  gap:8px;
  position:sticky;
  top:0;
  z-index:1000;
  height:{NAV_H}px;
  align-items:center;
  box-shadow:0 2px 8px rgba(0,0,0,.3);
}}
nav a {{
  color:#a8d8ea;
  text-decoration:none;
  padding:6px 16px;
  border-radius:20px;
  font-size:.88rem;
  font-weight:500;
  transition:background .2s, color .2s;
  white-space:nowrap;
}}
nav a:hover  {{ background:rgba(255,255,255,.15); color:#fff; }}
nav a.active {{ background:rgba(255,255,255,.2);  color:#fff; }}

/* -- Scroll offset so sticky nav doesn't cover section headings -- */
section {{
  scroll-margin-top:{NAV_H + 16}px;
  background:#fff;
  border-radius:12px;
  padding:32px;
  margin-bottom:28px;
  box-shadow:0 2px 12px rgba(0,0,0,.08);
}}

.container {{ max-width:1200px; margin:32px auto; padding:0 24px; }}

h2 {{ font-size:1.35rem; color:#0f3460; margin-bottom:18px;
      border-bottom:2px solid #e0e6f0; padding-bottom:8px; }}
h3 {{ font-size:1.1rem; color:#16213e; margin:20px 0 10px; }}

/* -- Metric cards -- */
.metric-grid {{ display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }}
.metric-card {{ background:#f0f4ff; border-radius:8px; padding:14px 20px;
                text-align:center; min-width:140px; flex:1; }}
.metric-val  {{ font-size:1.5rem; font-weight:700; color:#0f3460; }}
.metric-lbl  {{ font-size:.78rem; color:#555; margin-top:4px; }}

/* -- Cluster cards -- */
.cluster-card {{ border:1px solid #e0e6f0; border-radius:8px;
                 padding:16px; margin:12px 0; }}
.cluster-card h4 {{ color:#0f3460; margin-bottom:8px; font-size:1rem; }}
.kw-chip {{ display:inline-block; background:#e8f4f8; color:#0f3460;
            border-radius:12px; padding:2px 10px; font-size:.8rem; margin:2px; }}
.rep-title {{ background:#fff8e1; border-left:3px solid #e9c46a;
              padding:6px 12px; margin:4px 0; font-size:.88rem;
              border-radius:0 4px 4px 0; }}
.summary-text {{ font-size:.88rem; color:#555; margin:8px 0;
                 font-style:italic; }}

/* -- Table -- */
table {{ width:100%; border-collapse:collapse; font-size:.88rem; margin-top:12px; }}
th {{ background:#0f3460; color:#fff; padding:10px 14px; text-align:left; }}
td {{ padding:9px 14px; border-bottom:1px solid #eee; }}
tr:hover td {{ background:#f5f8ff; }}

/* -- Badges -- */
.badge {{ display:inline-block; padding:2px 8px; border-radius:10px;
          font-size:.75rem; font-weight:600; }}
.badge-green  {{ background:#d4edda; color:#155724; }}
.badge-blue   {{ background:#cce5ff; color:#004085; }}
.badge-orange {{ background:#fff3cd; color:#856404; }}

/* -- Method divider -- */
.method-block {{ margin-top:24px; border-top:2px dashed #e0e6f0; padding-top:16px; }}

footer {{ text-align:center; padding:32px; color:#888; font-size:.85rem; }}
img {{ max-width:100%; display:block; }}
"""

    js = f"""
// Highlight nav link for the section currently in view
(function() {{
  const NAV_H = {NAV_H};
  const links = document.querySelectorAll('nav a');
  const sections = document.querySelectorAll('section[id]');

  function activate() {{
    let current = '';
    sections.forEach(sec => {{
      if (window.scrollY >= sec.offsetTop - NAV_H - 20) current = sec.id;
    }});
    links.forEach(a => {{
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    }});
  }}
  window.addEventListener('scroll', activate, {{ passive: true }});
  activate();
}})();
"""

    # -- Corpus stats --------------------------------------------------------
    import numpy as np
    n_docs   = len(df)
    avg_len  = int(df["text"].str.split().str.len().mean())
    cpc_counts = df["cpc_code"].value_counts().to_dict() if "cpc_code" in df.columns else {}
    try:
      vocab_size = len(features["tfidf_vec"].get_feature_names_out())
    except Exception:
        vocab_size = "N/A"

    # Compute ROUGE
    avg_r1 = avg_r2 = avg_rl = None
    if summary_scores:
        avg_r1 = np.mean([v["rouge1"] for v in summary_scores.values()])
        avg_r2 = np.mean([v["rouge2"] for v in summary_scores.values()])
        avg_rl = np.mean([v["rougeL"] for v in summary_scores.values()])

    # Build metric cards
    metric_cards = [
        _metric_card("Total Patents", n_docs),
        _metric_card("Avg. Token Length", avg_len),
        _metric_card("CPC Classes", len(cpc_counts)),
        _metric_card("Vocabulary (TF-IDF)", vocab_size),
        _metric_card("LDA Coherence", features.get("lda_coherence", "N/A")),
    ]

    # Add ROUGE
    if avg_r1 is not None:
        avg_cov = np.mean([v["coverage"] for v in summary_scores.values()])
        avg_prox = np.mean([v["proximity"] for v in summary_scores.values()])

        metric_cards.extend([
            _metric_card("ROUGE-1", avg_r1),
            _metric_card("ROUGE-2", avg_r2),
            _metric_card("ROUGE-L", avg_rl),
            _metric_card("Coverage", avg_cov),
            _metric_card("Centroid Sim", avg_prox),
        ])

    # Final HTML
    corpus_html = f"""
    <div class="metric-grid">
      {''.join(metric_cards)}
    </div>

    <h3>CPC Class Distribution</h3>
    <table>
      <tr><th>CPC Code</th><th>Patent Count</th></tr>
      {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in cpc_counts.items())}
    </table>
    """
    # -- Metrics table --------------------------------------------------------
    metrics_rows = ""
    for method, res in cluster_results.items():
        m    = res["metrics"]
        sil  = m.get("silhouette", float("nan"))
        db   = m.get("davies_bouldin", float("nan"))
        stab = m.get("stability_ari", float("nan"))
        ari  = m.get("ari", None)
        nmi  = m.get("nmi", None)
        sb   = "badge-green" if sil  > 0.3  else "badge-orange"
        db_b = "badge-green" if db   < 1.0  else "badge-orange"
        stb  = "badge-green" if stab > 0.7  else "badge-blue"
        metrics_rows += f"""
        <tr>
          <td><strong>{method}</strong></td>
          <td><span class="badge {sb}">{sil:.4f}</span></td>
          <td><span class="badge {db_b}">{db:.4f}</span></td>
          <td><span class="badge {stb}">{stab:.4f}</span></td>
          <td>{'--' if ari is None else f'{ari:.4f}'}</td>
          <td>{'--' if nmi is None else f'{nmi:.4f}'}</td>
        </tr>"""

    # -- Cluster summaries ----------------------------------------------------
    cluster_html = ""
    for method_name, summaries in all_summaries.items():
        cluster_html += f'<div class="method-block"><h3>📌 {method_name}</h3>'
        for s in summaries:
            chips   = "".join(f'<span class="kw-chip">{k}</span>'
                               for k in s["keywords"][:7])
            rep_div = "".join(f'<div class="rep-title">* {t}</div>'
                               for t in s["representative_titles"])
            cluster_html += f"""
            <div class="cluster-card">
              <h4>Cluster {s['cluster_id']} &nbsp;-&nbsp; {s['size']} patents</h4>
              <div style="margin-bottom:8px">{chips}</div>
              <p class="summary-text">{s['summary'][:320]} ...</p>
              {rep_div}
            </div>"""
        cluster_html += "</div>"

    img = lambda name: _img_tag(os.path.join(OUT_DIR, name))

    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    lda_dim   = features["lda"].shape[1]
    sbert_dim = features["sbert"].shape[1]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patent Corpus Analysis Report</title>
<style>{css}</style>
</head>
<body>

<header>
  <h1>Patent Corpus Semantic Analysis</h1>
  <p>TF-IDF &nbsp;-&nbsp; LDA &nbsp;-&nbsp; Sentence-BERT &nbsp;-&nbsp;
     K-Means &nbsp;-&nbsp; t-SNE &nbsp;|&nbsp; Generated {now}</p>
  <p>Group 22 - Project 9 - CSE 573 Spring 2026 - Arizona State University</p>
</header>

<nav>
  <a href="#corpus">Corpus</a>
  <a href="#features">Features</a>
  <a href="#clustering">Clustering</a>
  <a href="#summaries">Summaries</a>
  <a href="#visualizations">Visualizations</a>
</nav>

<div class="container">

  <!-- -- 1. Corpus -- -->
  <section id="corpus">
    <h2>1 - Corpus Overview</h2>
    {corpus_html}
  </section>

  <!-- -- 2. Features -- -->
  <section id="features">
    <h2>2 - Feature Engineering &amp; Topic Modeling</h2>
    <p>Three parallel document representations were built:</p>
    <ul style="margin:12px 0 12px 22px;line-height:2.2;">
      <li><strong>TF-IDF + LSA</strong> -- sparse lexical vectors reduced via SVD</li>
      <li><strong>LDA</strong> -- document-topic probability distributions
          ({lda_dim} topics)</li>
      <li><strong>Sentence-BERT</strong> -- dense contextual embeddings
          ({sbert_dim}-dim)</li>
    </ul>
    <h3>LDA Topic Word Distributions</h3>
    {img("lda_topics.png")}
    <h3>Document-Topic Probability Heatmap</h3>
    {img("doc_topic_heatmap.png")}
  </section>

  <!-- -- 3. Clustering -- -->
  <section id="clustering">
    <h2>3 - Clustering Evaluation</h2>
    <p>K-Means (k-means++ init) applied to all three representations.
       <strong>Silhouette ↑ better - Davies-Bouldin ↓ better - Stability ↑ better.</strong></p>
    <table>
      <tr>
        <th>Method</th>
        <th>Silhouette ↑</th>
        <th>Davies-Bouldin ↓</th>
        <th>Stability (ARI) ↑</th>
        <th>Ext. ARI</th>
        <th>NMI</th>
      </tr>
      {metrics_rows}
    </table>
    <br>
    {img("metrics_comparison.png")}
    {img("cluster_sizes.png")}
    <h3>Optimal k Selection (SBERT)</h3>
    {img("optimal_k_SBERT.png")}
  </section>

  <!-- -- 4. Summaries -- -->
  <section id="summaries">
    <h2>4 - Cluster Summaries</h2>
    <p>Each cluster shows its top keywords, an extractive summary,
       and the most representative patent titles (closest to centroid).</p>
    {cluster_html}
  </section>

  <!-- -- 5. Visualizations -- -->
  <section id="visualizations">
    <h2>5 - Dimensionality Reduction Visualizations</h2>

    <h3>All Three Methods -- t-SNE Side by Side</h3>
    {img("all_projections_tsne.png")}

    <h3>All Three Methods -- PCA Side by Side</h3>
    {img("all_projections_pca.png")}

    <h3>TF-IDF + K-Means (t-SNE)</h3>
    {img("TF-IDF_+_KMeans_tsne.png")}

    <h3>TF-IDF + K-Means (PCA)</h3>
    {img("TF-IDF_+_KMeans_pca.png")}

    <h3>LDA + K-Means (t-SNE)</h3>
    {img("LDA_+_KMeans_tsne.png")}

    <h3>LDA + K-Means (PCA)</h3>
    {img("LDA_+_KMeans_pca.png")}

    <h3>SBERT + K-Means (t-SNE)</h3>
    {img("SBERT_+_KMeans_tsne.png")}

    <h3>SBERT + K-Means (PCA)</h3>
    {img("SBERT_+_KMeans_pca.png")}
  </section>

</div>

<footer>
  Patent Semantic Analysis Pipeline &nbsp;-&nbsp;
  TF-IDF - LDA - Sentence-BERT - K-Means - t-SNE &nbsp;-&nbsp;
  Group 22 - CSE 573 Spring 2026 - Arizona State University
</footer>

<script>{js}</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] Saved -> {output_path}")
    return output_path
