#!/usr/bin/env python
"""Generate PLACEHOLDER data in the official format so the pipeline runs end-to-end before
the real dataset lands. Stdlib only. Numbers produced on this data mean nothing.

    python scripts/make_fake_data.py --out data/fake_dataset
    export BER_DATA_DIR=$PWD/data/fake_dataset      # (PowerShell: $env:BER_DATA_DIR="...")
"""
import argparse
import csv
import random
from pathlib import Path

WORDS = ["sun", "star", "global", "apex", "blue", "river", "tech", "om", "sai", "shree", "metro", "prime", "golden",
         "united", "royal", "green", "city", "national", "eagle", "lotus", "krishna", "pacific", "summit", "harbor",
         "bharat", "vista", "nova", "delta", "crown", "silver", "maple", "ganesh", "liberty", "orient", "coastal"]
KINDS = ["traders", "foods", "motors", "pharma", "textiles", "solutions", "systems", "electricals", "logistics",
         "bakery", "hardware", "clinic", "consulting", "builders", "exports"]
COUNTRIES = {
    "US": {"legal": ["Inc", "LLC", "Corp", "Co"], "streets": ["Main St", "Oak Ave", "Market Street", "Elm Rd", "Broadway"],
           "cities": ["Austin, TX", "Denver, CO", "Seattle, WA", "Columbus, OH"], "post": lambda r: f"{r.randint(10000, 99999)}"},
    "India": {"legal": ["Pvt Ltd", "Private Limited", "LLP", "Ltd"], "streets": ["MG Road", "Station Rd", "Nehru Nagar", "Gandhi Marg"],
              "cities": ["Pune, Maharashtra", "Jaipur, Rajasthan", "Hyderabad, Telangana", "Chennai, Tamil Nadu"],
              "post": lambda r: f"{r.randint(110000, 859999)}"},
    "France": {"legal": ["SARL", "SAS", "SA"], "streets": ["rue de la Paix", "avenue Victor Hugo", "bd Saint-Michel"],
               "cities": ["Lyon", "Paris", "Marseille", "Toulouse"], "post": lambda r: f"{r.randint(10000, 95999)}"},
}
ABBR = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Private": "Pvt", "Limited": "Ltd", "Corporation": "Corp", "and": "&"}


def base_entity(r, country):
    c = COUNTRIES[country]
    name = f"{r.choice(WORDS).title()} {r.choice(WORDS).title()} {r.choice(KINDS).title()} {r.choice(c['legal'])}"
    addr = f"{r.randint(1, 999)} {r.choice(c['streets'])}, {r.choice(c['cities'])} {c['post'](r)}"
    return name, addr


def noisy(r, text):
    toks = text.split()
    op = r.random()
    if op < 0.25 and len(toks) > 2:
        toks = toks[:-1]                                   # drop suffix / component
    elif op < 0.45:
        toks = [ABBR.get(t, t) for t in toks]              # abbreviate
    elif op < 0.6 and len(toks) > 2:
        i = r.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]        # transposition
    s = " ".join(toks)
    if r.random() < 0.4 and len(s) > 4:                    # typo
        i = r.randrange(len(s) - 1)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if r.random() < 0.3:
        s = s.upper() if r.random() < 0.5 else s.lower()
    return s


def make_part(r, part, countries, n_s1):
    s1, s2, s3, gt = [], [], [], []
    n2 = n3 = 0
    for i in range(1, n_s1 + 1):
        country = r.choice(countries)
        name, addr = base_entity(r, country)
        sid = f"S1-{i:05d}"
        s1.append([sid, name, addr, country])
        k = r.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
        matched = []
        for _ in range(k):
            if r.random() < 0.5:
                n2 += 1; rid = f"S2-{n2:05d}"; s2.append([rid, noisy(r, name), noisy(r, addr), country])
            else:
                n3 += 1; rid = f"S3-{n3:05d}"; s3.append([rid, noisy(r, name), noisy(r, addr) if r.random() < 0.8 else "", country])
            matched.append(rid)
        gt.append([sid, ",".join(matched)])
        if r.random() < 0.3:  # hard negative: same name, different address
            n2 += 1; s2.append([f"S2-{n2:05d}", noisy(r, name), base_entity(r, country)[1], country])
    for _ in range(n_s1 // 2):  # unrelated distractors
        country = r.choice(countries)
        name, addr = base_entity(r, country)
        n3 += 1; s3.append([f"S3-{n3:05d}", name, addr, country])
    r.shuffle(s2); r.shuffle(s3)
    return s1, s2, s3, gt


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_dataset")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=600)
    a = ap.parse_args()
    r = random.Random(0)
    out = Path(a.out)
    hdr = ["entity_id", "business_name", "business_address", "country"]
    for part, countries, n in (("train", ["US", "India"], a.n_train), ("test", ["US", "India", "France"], a.n_test)):
        s1, s2, s3, gt = make_part(r, part, countries, n)
        write(out / part / f"{part}_source1.tsv", hdr, s1)
        write(out / part / f"{part}_source2.tsv", hdr, s2)
        write(out / part / f"{part}_source3.tsv", hdr, s3)
        if part == "train":
            write(out / part / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    print(f"fake dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
