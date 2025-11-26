import pandas as pd
import json
import math
import html
import unicodedata
from tqdm.auto import tqdm
import datetime

# ------------------------------------------------------------
# CONFIG — change these paths to your actual dataset locations
# ------------------------------------------------------------

METADATA_PATH = "/Users/prajwal/Desktop/JSON/AI Capstone/Product-Management-AI-Agent/Data/meta_Beauty_and_Personal_Care.jsonl"        # your big metadata file (JSONL)
REVIEWS_PATH = "/Users/prajwal/Desktop/JSON/AI Capstone/Product-Management-AI-Agent/Data/Beauty_and_Personal_Care.jsonl"          # your big reviews file (JSONL)
# OUTPUT_PATH = "/Users/prajwal/Desktop/JSON/AI Capstone/Product-Management-AI-Agent/Data/merged_reviews.jsonl"   # output file
PRODUCTS_OUT = "/Users/prajwal/Desktop/JSON/AI Capstone/Product-Management-AI-Agent/Data/products_clean.jsonl"
REVIEWS_OUT = "/Users/prajwal/Desktop/JSON/AI Capstone/Product-Management-AI-Agent/Data/reviews_clean.jsonl"

# Adjust chunksizes if needed
META_CHUNK = 2500
REVIEWS_CHUNK = 2500

# -----------------------------
# TEXT & VALUE CLEANING HELPERS
# -----------------------------

SMART_CHAR_MAP = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",
}

beauty_keywords = [
    "serum", "moisturizer", "cleanser", "toner", "essence",
    "exfoliant", "face wash", "gel cream", "cream", "lotion",
    "peptide", "retinol", "vitamin c", "niacinamide",
    "hyaluronic", "spf", "sunscreen", "skin"
]

def clean_text(text: str | None) -> str | None:
    """Normalize weird characters, HTML entities, and whitespace."""
    if not isinstance(text, str):
        return None

    # Decode HTML entities (&amp;, &quot;, etc.)
    text = html.unescape(text)

    # Replace smart quotes / dashes with ASCII equivalents
    for bad, good in SMART_CHAR_MAP.items():
        text = text.replace(bad, good)

    # Normalize unicode (e.g. full-width chars)
    text = unicodedata.normalize("NFKC", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Optional: collapse very long whitespace runs
    text = " ".join(text.split())

    return text or None


def clean_value(v):
    """Convert NaN -> None, leave other types as-is."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def clean_list(v):
    """Ensure we always get a list (for categories, features, etc.)."""
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    return [v]

def is_beauty_main_category(cat):
    cat = clean_value(cat)
    if not isinstance(cat, str):
        return False
    return "beauty" in cat.lower()

def is_beauty_categories_list(categories):
    categories = clean_list(categories)
    return any("skincare" in str(c.lower()) for c in categories) or any("skin care" in str(c.lower()) for c in categories)

def title_has_keyword(title):
    title = clean_value(title)
    if not isinstance(title, str):
        return False
    t = title.lower()
    return any(k in t for k in beauty_keywords)

def clean_timestamp(v):
    """Convert pandas Timestamp or datetime to ISO string. Keep ints as-is. Convert NaN to None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None

    # if already int (UNIX), keep it
    if isinstance(v, int):
        return v

    # if pandas Timestamp or datetime object
    if isinstance(v, (pd.Timestamp, datetime.datetime)):
        return v.isoformat()

    # if it's a string that looks like a date, keep it
    if isinstance(v, str):
        return v

    return None  # fallback

# -----------------------------
# 1. PROCESS METADATA → products_clean.jsonl
# -----------------------------

print("\nSTEP 1: Processing metadata → products_clean.jsonl")

products_seen = set()
products_file = open(PRODUCTS_OUT, "w", encoding="utf-8")

meta_iter = pd.read_json(METADATA_PATH, lines=True, chunksize=META_CHUNK)

for chunk in tqdm(meta_iter, desc="Metadata chunks"):

    for _, row in chunk.iterrows():

        # Prefer parent_asin if available, else asin
        asin = clean_value(row.get("asin"))
        parent_asin = clean_value(row.get("parent_asin"))
        product_key = parent_asin or asin

        if not product_key:
            continue

        if product_key in products_seen:
            continue  # avoid duplicates

        products_seen.add(product_key)

        # Extract brand from multiple possible places
        details = row.get("details", {}) or {}
        raw_brand = (
            row.get("brand")
            or row.get("Brand")
            or details.get("brand")
            or details.get("Brand")
        )

        product = {
            "asin": product_key,
            "product_name": clean_text(clean_value(row.get("title"))),
            "brand": clean_text(clean_value(raw_brand)),
            "main_category": clean_text(clean_value(row.get("main_category"))),
            "categories": clean_list(row.get("categories")),
            "price": clean_value(row.get("price")),
            "average_rating": clean_value(row.get("average_rating")),
            "rating_number": clean_value(row.get("rating_number")),
            "features": [clean_text(f) for f in clean_list(row.get("features")) if clean_text(f)],
            "description": [clean_text(d) for d in clean_list(row.get("description")) if clean_text(d)],
            "store": clean_text(clean_value(row.get("store"))),
        }

        # Apply BEAUTY filters
        is_beauty = (
            is_beauty_main_category(product["main_category"])
            and is_beauty_categories_list(product["categories"])
        )
        if not is_beauty:
            continue

        # Only keep products that have at least a name or description
        if not product["product_name"] and not product["description"]:
            continue

        products_file.write(json.dumps(product, ensure_ascii=False) + "\n")

products_file.close()
print(f"Products written: {len(products_seen):,} → {PRODUCTS_OUT}")


# -----------------------------
# 2. BUILD product_map (asin → small info) FOR REVIEW ENRICHMENT
# -----------------------------

print("\nSTEP 2: Building product_map from products_clean.jsonl")

product_map: dict[str, dict] = {}

with open(PRODUCTS_OUT, "r", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        asin = p["asin"]

        product_map[asin] = {
            "product_name": p.get("product_name"),
            "brand": p.get("brand"),
            "main_category": p.get("main_category"),
            "categories": p.get("categories") or [],
        }

print(f"Products in product_map: {len(product_map):,}")


# -----------------------------
# 3. PROCESS REVIEWS → reviews_clean.jsonl
# -----------------------------

print("\nSTEP 3: Processing reviews → reviews_clean.jsonl")

reviews_file = open(REVIEWS_OUT, "w", encoding="utf-8")

review_iter = pd.read_json(REVIEWS_PATH, lines=True, chunksize=REVIEWS_CHUNK)

total_reviews = 0
kept_reviews = 0
skipped_no_product = 0
skipped_empty_text = 0

for chunk in tqdm(review_iter, desc="Review chunks"):

    for _, r in chunk.iterrows():
        total_reviews += 1

        asin = clean_value(r.get("asin"))
        if not asin:
            skipped_no_product += 1
            continue

        product = product_map.get(asin)
        if product is None:
            # review for product we did not keep
            skipped_no_product += 1
            continue

        # --- Handle different possible review schemas ---

        # rating: either "rating" or "overall"
        rating = clean_value(r.get("rating"))
        if rating is None:
            rating = clean_value(r.get("overall"))

        # review text: either "text" or "reviewText"
        raw_text = r.get("text")
        if raw_text is None:
            raw_text = r.get("reviewText")

        review_text = clean_text(raw_text)

        # title/summary
        raw_title = r.get("title")
        if raw_title is None:
            raw_title = r.get("summary")
        review_title = clean_text(raw_title)

        # verified flag: "verified_purchase" or "verified"
        verified = r.get("verified_purchase")
        if verified is None:
            verified = r.get("verified")
        if isinstance(verified, float) and math.isnan(verified):
            verified = None

        # help votes: "helpful_vote" or "vote"
        helpful = clean_value(r.get("helpful_vote"))
        if helpful is None:
            helpful = clean_value(r.get("vote"))
        # some datasets store "vote" as string "12" or "1,234"
        if isinstance(helpful, str):
            try:
                helpful = int(helpful.replace(",", ""))
            except Exception:
                helpful = None
        helpful = helpful or 0

        # timestamp: either "timestamp" or "unixReviewTime"
        ts = clean_timestamp(r.get("timestamp"))

        # reviewer id
        user_id = clean_value(r.get("user_id") or r.get("reviewerID"))

        # Skip reviews with no meaningful text
        if not review_text or len(review_text) < 10:
            skipped_empty_text += 1
            continue

        review = {
            "review_id": f"{asin}_{user_id}_{ts}",
            "asin": asin,
            "rating": rating,
            "title": review_title,
            "text": review_text,
            "timestamp": ts,
            "verified_purchase": verified,
            "helpful_vote": helpful,
            "user_id": user_id,

            # Light product linkage
            "product_name": product["product_name"],
            "brand": product["brand"],
            "main_category": product["main_category"],
            "categories": product["categories"],
        }

        reviews_file.write(json.dumps(review, ensure_ascii=False) + "\n")
        kept_reviews += 1

reviews_file.close()

print("\nDONE ✅")
print(f"Total raw reviews seen: {total_reviews:,}")
print(f"Reviews kept (cleaned): {kept_reviews:,}")
print(f"Skipped (no product match): {skipped_no_product:,}")
print(f"Skipped (empty/too short text): {skipped_empty_text:,}")
print(f"Clean products → {PRODUCTS_OUT}")
print(f"Clean reviews  → {REVIEWS_OUT}")