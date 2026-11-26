"""Build data/kaggle_products.csv from the Fragrantica Kaggle archive (data/raw/fra_perfumes.csv).

One row per perfume: brand, name, gender, average rating (1-5), number of ratings, top-5 main
accords, and the top/middle/base notes parsed from the description. There is NO per-user data
in this source, so models on it are product-level only.
"""
import ast, re
from pathlib import Path
import pandas as pd

RAW = Path(__file__).parent / "raw" / "fra_perfumes.csv"
OUT = Path(__file__).parent / "kaggle_products.csv"


def notes(desc, which):
    m = re.search(which + r" notes are (.*?)(?:;|\.$|\. )", str(desc))
    return m.group(1).replace(" and ", ", ") if m else ""


def main():
    fra = pd.read_csv(RAW)
    x = fra.url.str.extract(r"perfume/([^/]+)/(.+)-\d+\.html")
    df = pd.DataFrame({
        "brand": x[0].str.lower(), "perfume": x[1].str.lower(),
        "gender": fra.Gender.str.replace("for ", "", regex=False),
        "rating": pd.to_numeric(fra["Rating Value"], errors="coerce"),
        "rating_count": pd.to_numeric(fra["Rating Count"].astype(str).str.replace(",", ""), errors="coerce"),
        "accords": fra["Main Accords"].map(lambda s: ";".join(ast.literal_eval(s)[:5]) if isinstance(s, str) else ""),
        "top_notes": fra.Description.map(lambda d: notes(d, "Top")),
        "middle_notes": fra.Description.map(lambda d: notes(d, "middle")),
        "base_notes": fra.Description.map(lambda d: notes(d, "base")),
        "url": fra.url,
    }).dropna(subset=["brand", "rating", "rating_count"]).drop_duplicates("url")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} perfumes")


if __name__ == "__main__":
    main()
