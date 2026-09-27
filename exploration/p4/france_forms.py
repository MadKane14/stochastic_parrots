"""T23b (p4): label-free counts of France TEST address/name forms (no labels). Share of S1 vs S2/S3 records with each form.
Usage (repo root): python exploration/p4/france_forms.py > exploration/p4/france_forms_out.md"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from france_audit import load_country
recs = load_country(Path(__file__).resolve().parents[2] / "data/dataset/test", "France")
s1 = recs["source"] == 1
A, N = recs["business_address"], recs["business_name"]
pats = {
 "addr: department name (Nord|Gironde|Pas-de-Calais|Loire-Atlantique|Vendée|...)": (A, r"\b(Nord|Gironde|Pas-de-Calais|Loire-Atlantique|Vend[ée]e|Maine-et-Loire|Sarthe|Mayenne|Landes|Dordogne|Lot-et-Garonne|Pyr[ée]n[ée]es-Atlantiques)\b"),
 "addr: region name (Hauts-de-France|Pays de la Loire|Nouvelle-Aquitaine)": (A, r"Hauts-de-France|Pays de la Loire|Nouvelle-Aquitaine"),
 "addr: department number in parentheses '(59)'": (A, r"\(\d{2,3}\)"),
 "addr: number-dash-street '5 - Rue'": (A, r"\b\d+\s*-\s+\w"),
 "addr: 'bis/ter' word": (A, r"\b\d+\s+(bis|ter|quater)\b"),
 "addr: letter suffix '6B' / '6 B'": (A, r"\b\d+\s?[A-D]\b"),
 "addr: 'N°'": (A, r"N°|\bNo\b\.?"),
 "addr: floor/apt ('etage','appt','apt','bat')": (A, r"(?i)\b(e|é)tage\b|\bappt?\b|\bbat\b|\bbâtiment\b"),
 "addr: 5-digit postcode": (A, r"\b\d{5}\b"),
 "name: legal form SARL/EURL/SAS/SASU/SCI/SNC/EI/SA": (N, r"(?i)\b(sarl|eurl|sasu|sas|sci|snc|ei|sa)\b"),
 "name: dotted legal form 'S.A.R.L.'/'S.A.S.'": (N, r"(?i)\bs\.\s?a\.(\s?r\.\s?l\.|\s?s\.)"),
 "name: '(France)'": (N, r"(?i)\(france\)"),
 "name: web-style (.com/.fr/www)": (N, r"(?i)\.(com|fr|net)\b|www\."),
 "name: acronym-only (2-4 capitals, maybe + legal form)": (N, r"^\s*[A-Z]{2,4}(\s+(SARL|SAS|SASU|EURL|SCI|SNC|SA))?\s*$"),
 "name: 'dba'/'formerly'/'aka'": (N, r"(?i)\b(dba|formerly|aka|doing business)\b"),
}
print("| form | S1 records | share S1 | S2/S3 records | share S2/S3 |\n|---|---|---|---|---|")
for k, (col, p) in pats.items():
    m = col.str.contains(p, regex=True)
    print(f"| {k} | {int(m[s1].sum()):,} | {m[s1].mean():.2%} | {int(m[~s1].sum()):,} | {m[~s1].mean():.2%} |")
