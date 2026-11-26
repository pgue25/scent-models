"""Build data/amazon_clean.csv from the public Amazon Reviews 2023 files + the Fragrantica Kaggle file.

Steps (each was checked by hand; see data/README.md for counts):
  1. Fragrance products: Beauty_and_Personal_Care + All_Beauty metadata, category contains "Fragrance".
  2. Their reviews, de-duplicated on (reviewer, product).
  3. Scent filter: keep reviews that talk about the smell; drop ones about shipping, price,
     damage or fakes; drop reviewers who gave every product the same stars.
  4. Match products to Fragrantica (brand in title, then longest perfume name). Unmatched dropped.
  5. Keep reviewers with 7+ remaining reviews. Split each person's ratings 70/30 (seed 0).

    python data/build_amazon.py            # downloads ~3.7 GB once into data/raw/ (streamed, filtered)

Needs data/raw/fra_perfumes.csv from the Kaggle archive.
"""
import gzip, json, re, subprocess, sys, unicodedata, collections
from pathlib import Path
import numpy as np, pandas as pd

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "amazon_clean.csv"
BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
SMELL = r"\bsmell|\bscent|\bnotes?\b|\bfragran|\bsweet|\blast(?:s|ed|ing)?\b|longevity|projection|sillage|\bmusky?\b|\bfloral|\bfruity|\bvanilla|\bcitrus|\bwoody|\bpowdery|\bspicy|\bfresh\b|\bstinks?\b|\bodor|\baroma|\bnose\b|\bwears? off|\bfades?\b|\bdry ?down"
JUNK = r"\bship|\bdeliver|\barriv|\bpackag|\bbox\b|\bprice\b|\bexpensive|\bcheap\b|\bbroken\b|\bleak|\bdamag|\bfake\b|\bcounterfeit|\bknock ?off|\bnot (?:the )?(?:real|authentic|genuine)|\bwatered[- ]down|\brefund|\breturn(?:ed)?\b|\bseller\b|\bmissing\b|\bcap\b|\bsprayer|\bnozzle|\bpump\b"
MIN_REVIEWS, TEST_SHARE, SEED = 7, 0.3, 0


def stream(name):
    """Yield JSON lines from a remote .jsonl.gz without saving the whole file."""
    p = subprocess.Popen(f"curl -sL {BASE}/{name} | gunzip", shell=True, stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        yield json.loads(line)


def fragrance_products():
    path = RAW / "amazon_fragrance_meta.jsonl"
    if not path.exists():
        with open(path, "w") as out:
            for cat in ("Beauty_and_Personal_Care", "All_Beauty"):
                for d in stream(f"meta_categories/meta_{cat}.jsonl.gz"):
                    cats = " > ".join(d.get("categories") or []).lower()
                    title = (d.get("title") or "").lower()
                    if "fragrance" in cats or (not cats and re.search("eau de parfum|eau de toilette|cologne|perfume", title)):
                        out.write(json.dumps({k: d.get(k) for k in ("parent_asin", "title", "store")}) + "\n")
    return pd.read_json(path, lines=True).drop_duplicates("parent_asin")


def fragrance_reviews(asins):
    path = RAW / "amazon_fragrance_reviews.jsonl"
    if not path.exists():
        with open(path, "w") as out:
            for cat in ("Beauty_and_Personal_Care", "All_Beauty"):
                for d in stream(f"review_categories/{cat}.jsonl.gz"):
                    if d["parent_asin"] in asins:
                        out.write(json.dumps({k: d[k] for k in ("user_id", "parent_asin", "rating", "title", "text")}) + "\n")
    return pd.read_json(path, lines=True).drop_duplicates(["user_id", "parent_asin"])


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", s).split()) + " "


def match_fragrantica(products):
    fra = pd.read_csv(RAW / "fra_perfumes.csv")
    x = fra.url.str.extract(r"perfume/([^/]+)/(.+)-\d+\.html")
    fra["brand"], fra["perfume"] = x[0].str.lower(), x[1].str.lower()
    fra = fra.dropna(subset=["brand"]).drop_duplicates("url").reset_index(drop=True)
    fra["b"], fra["p"] = fra.brand.str.replace("-", " ").map(norm), fra.perfume.str.replace("-", " ").map(norm)
    by_brand = collections.defaultdict(list)
    for i, b in fra.b.items():
        by_brand[b].append(i)
    brands = sorted(by_brand, key=len, reverse=True)
    hits = {}
    for asin, title, store in zip(products.parent_asin, products.title.map(norm), products.store.map(norm)):
        best, best_len = None, 0
        for b in brands:
            if len(b) > 4 and (b in title or b == store or (len(b) > 6 and b in store)):
                rest = title.replace(b, " ")
                for i in by_brand[b]:
                    p = fra.p[i]
                    if len(p) > 3 and p in rest and len(p) > best_len:
                        best, best_len = i, len(p)
        if best is not None:
            hits[asin] = best
    return fra, pd.Series(hits, name="fra_row")


def main():
    RAW.mkdir(exist_ok=True)
    products = fragrance_products()
    rev = fragrance_reviews(set(products.parent_asin))
    t = (rev.title.fillna("") + " " + rev.text.fillna("")).str.lower()
    rev = rev[t.str.contains(SMELL) & ~t.str.contains(JUNK)]
    n = rev.groupby("user_id").parent_asin.transform("nunique")
    rev = rev[n >= 3]
    spread = rev.groupby("user_id").rating.transform(lambda x: x.max() - x.min())
    rev = rev[spread > 0]

    fra, hit = match_fragrantica(products[products.parent_asin.isin(rev.parent_asin)])
    rev = rev[rev.parent_asin.isin(hit.index)].copy()
    rev = rev[rev.groupby("user_id").parent_asin.transform("size") >= MIN_REVIEWS].reset_index(drop=True)
    f = fra.loc[hit[rev.parent_asin].to_numpy()].reset_index(drop=True)
    rev["brand"], rev["perfume"], rev["gender"] = f.brand, f.perfume, f.Gender
    rev["fra_rating"] = pd.to_numeric(f["Rating Value"], errors="coerce")
    rev["fra_count"] = pd.to_numeric(f["Rating Count"].astype(str).str.replace(",", ""), errors="coerce")
    rev["accords"] = f["Main Accords"].map(lambda s: ";".join(eval(s)[:5]) if isinstance(s, str) else "")

    rng = np.random.default_rng(SEED)
    rev["split"] = "train"
    for _, g in rev.groupby("user_id"):
        rev.loc[rng.permutation(g.index)[:round(TEST_SHARE * len(g))], "split"] = "test"
    cols = ["user_id", "parent_asin", "split", "rating", "brand", "perfume", "gender", "fra_rating", "fra_count", "accords"]
    rev[cols].to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(rev)} ratings, {rev.user_id.nunique()} people, {rev.parent_asin.nunique()} products")


if __name__ == "__main__":
    main()
