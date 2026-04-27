"""
Export 10,000 BigPatent records to DATA/real_patents_10000.csv

Runs the existing ingestion pipeline (streaming) to materialize a 10k CSV
for quick reuse and sharing. Requires the `datasets` package.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'CODE'))

from ingestion import load_corpus


def main():
    out_dir = os.path.join(ROOT, 'DATA')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'real_patents_10000.csv')

    print('[Export] Loading BigPatent (n=10000) ...')
    df = load_corpus('bigpatent', n=10000)
    print(f'[Export] Loaded {len(df)} records')

    cols = ['patent_id', 'title', 'abstract', 'cpc_code', 'text']
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f'Missing expected column: {c}')

    df.to_csv(out_csv, index=False)
    print(f'[Export] Wrote -> {out_csv}')


if __name__ == '__main__':
    main()
