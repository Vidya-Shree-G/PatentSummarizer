"""
Module 2: Text Preprocessing (stdlib + scikit-learn only)
"""
import re
import pandas as pd

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","dare","ought","used","this","that","these","those",
    "it","its","as","if","then","than","so","yet","both","either","each",
    "few","more","most","other","some","such","no","not","only","own","same",
    "too","very","just","because","while","although","however","into","through",
    "during","before","after","above","below","between","out","off","over",
    "under","again","further","once","here","there","when","where","why","how",
    "all","any","much","many","which","who","also","thus","well","therefore",
    "method","system","apparatus","device","invention","present","embodiment",
    "plurality","least","one","two","three","first","second","third","comprises",
    "comprising","configured","based","using","used","use","provide","includes",
    "including","wherein","thereby","thereof","therefor","herein",
    "described","disclosed","claim","claims",
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def simple_tokenize(text: str) -> list:
    return [t for t in text.split()
            if t.isalpha() and len(t) >= 3 and t not in STOPWORDS]

def preprocess_corpus(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    print("[Preprocessing] Cleaning & tokenising …")
    df["clean_text"] = df[text_col].apply(clean_text)
    df["tokens"]     = df["clean_text"].apply(simple_tokenize)
    df["processed"]  = df["tokens"].apply(lambda t: " ".join(t))
    df = df[df["processed"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"[Preprocessing] Done. {len(df)} documents.")
    return df
