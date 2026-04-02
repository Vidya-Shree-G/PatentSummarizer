# Patent Summarizer 🔬

End-to-end pipeline that clusters, summarizes and visualizes large patent corpora
using **TF-IDF · LDA · Sentence-BERT · K-Means · t-SNE**.

---

## Project Structure

```
PatentSummarizer/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py            ← pipeline entry point
├── ingestion.py       ← data loading (synthetic + PatentsView API)
├── preprocessing.py   ← text cleaning & tokenisation
├── features.py        ← TF-IDF/LSA, LDA, Sentence-BERT embeddings
├── clustering.py      ← K-Means, metrics, stability, optimal-k
├── summarization.py   ← extractive cluster summaries & keywords
├── visualization.py   ← t-SNE, PCA, topic bars, heatmaps, charts
└── report.py          ← self-contained HTML report generator
```

Outputs land in `./outputs/` on your host machine after every run.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) ≥ 2.0
  (included with Docker Desktop on Mac/Windows)

Verify with:

```bash
docker --version
docker compose version
```

---

## Quickstart (3 commands)

```bash
# 1. Clone / copy all project files into one folder, then enter it
cd PatentSummarizer

# 2. Build the image (only needed once, ~5 min first time)
docker compose build

# 3. Run the pipeline
docker compose up patent-pipeline
```

Results appear in `./outputs/`:

- `patent_analysis_report.html` — open in any browser
- `patent_clusters.csv`
- `*.png` — all 13 figures

---

## All Run Options

### Option A — Docker Compose (recommended)

```bash
# Synthetic corpus (fast, offline, 48 patents)
docker compose up patent-pipeline

# Real patents from PatentsView API (requires internet)
docker compose up patent-pipeline-real

# Quick smoke-test (no charts, no report — just verify it runs)
docker compose up patent-pipeline-test
```

### Option B — Plain `docker run`

```bash
# Build once
docker build -t patent-pipeline .

# Run with defaults
docker run --rm -v "$(pwd)/outputs:/app/outputs" patent-pipeline

# Custom parameters
docker run --rm -v "$(pwd)/outputs:/app/outputs" patent-pipeline \
  python main.py --source synthetic --n 48 --k 6 --topics 6

# Real patents, 500 docs, 8 clusters
docker run --rm -v "$(pwd)/outputs:/app/outputs" patent-pipeline \
  python main.py --source patentsview --n 500 --k 8 --topics 8

# Skip visualizations (faster)
docker run --rm -v "$(pwd)/outputs:/app/outputs" patent-pipeline \
  python main.py --no-viz --no-report
```

### Option C — Interactive shell inside the container

```bash
docker run --rm -it -v "$(pwd)/outputs:/app/outputs" patent-pipeline bash

# Then inside the container:
python main.py --source synthetic --n 48 --k 6
python ingestion.py        # test a single module
python features.py
exit
```

---

## CLI Arguments

| Argument      | Default     | Description                                  |
| ------------- | ----------- | -------------------------------------------- |
| `--source`    | `synthetic` | `synthetic` (offline) or `patentsview` (API) |
| `--n`         | `48`        | Number of patents to load                    |
| `--k`         | `6`         | Number of clusters                           |
| `--topics`    | `6`         | Number of LDA topics                         |
| `--no-viz`    | off         | Skip all chart generation                    |
| `--no-report` | off         | Skip HTML report                             |

---

## Outputs

| File                          | Description                               |
| ----------------------------- | ----------------------------------------- |
| `patent_analysis_report.html` | Full interactive report — open in browser |
| `patent_clusters.csv`         | Patents with cluster label per method     |
| `all_projections_tsne.png`    | Side-by-side t-SNE for all 3 methods      |
| `all_projections_pca.png`     | Side-by-side PCA for all 3 methods        |
| `TF-IDF_+_KMeans_tsne.png`    | TF-IDF cluster scatter                    |
| `LDA_+_KMeans_tsne.png`       | LDA cluster scatter                       |
| `SBERT_+_KMeans_tsne.png`     | SBERT cluster scatter                     |
| `metrics_comparison.png`      | Bar chart: Silhouette / DB / Stability    |
| `cluster_sizes.png`           | Patent count per cluster                  |
| `lda_topics.png`              | Top words per LDA topic                   |
| `doc_topic_heatmap.png`       | Document × Topic probability heatmap      |
| `optimal_k_SBERT.png`         | Silhouette + DB vs k line chart           |

---

## Rebuilding After Code Changes

If you edit any `.py` file:

```bash
docker compose build          # rebuild image
docker compose up patent-pipeline   # re-run
```

Or force a full rebuild with no cache:

```bash
docker compose build --no-cache
```

---

## Troubleshooting

**"permission denied" on outputs folder (Linux)**

```bash
mkdir -p outputs
chmod 777 outputs
docker compose up patent-pipeline
```

**Port / memory issues on Windows (Docker Desktop)**
In Docker Desktop → Settings → Resources, allocate at least **4 GB RAM**
(Sentence-BERT model download requires ~400 MB).

**PatentsView returns no results**
The API occasionally has downtime. The pipeline automatically falls back
to the synthetic corpus. Run with `--source synthetic` to bypass.

**Want to use your own patent CSV?**
Add a loader in `ingestion.py`:

```python
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["title"] + ". " + df["abstract"]
    return df
```

Then call `load_corpus(source="csv", path="your_file.csv")`.

---

## Pipeline Architecture

```
Raw Patents
    │
    ▼
[Ingestion]        USPTO / PatentsView API or synthetic corpus
    │
    ▼
[Preprocessing]    Clean → Tokenise → Remove stopwords
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[TF-IDF + LSA]                    [Sentence-BERT]
[LDA Topics]                       dense 384-dim
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
            [K-Means Clustering]
            (k=6, k-means++ init)
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
    [Metrics]  [Summaries] [Visualization]
    Silhouette  Keywords   t-SNE / PCA
    DB Index    Excerpts   Topic bars
    Stability             Heatmap
          │
          ▼
    [HTML Report]
```
