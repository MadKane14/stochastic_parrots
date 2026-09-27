/home/anishtoshniwal/stochastic_parrots/exploration/p4/france_forms.py:29: UserWarning: This pattern is interpreted as a regular expression, and has match groups. To actually get the groups, use str.extract.
  m = col.str.contains(p, regex=True)
| form | S1 records | share S1 | S2/S3 records | share S2/S3 |
|---|---|---|---|---|
| addr: department name (Nord|Gironde|Pas-de-Calais|Loire-Atlantique|Vendée|...) | 439 | 0.17% | 456,740 | 31.83% |
| addr: region name (Hauts-de-France|Pays de la Loire|Nouvelle-Aquitaine) | 259,452 | 100.00% | 478,124 | 33.32% |
| addr: department number in parentheses '(59)' | 0 | 0.00% | 18,716 | 1.30% |
| addr: number-dash-street '5 - Rue' | 264 | 0.10% | 24,276 | 1.69% |
| addr: 'bis/ter' word | 7,892 | 3.04% | 898 | 0.06% |
| addr: letter suffix '6B' / '6 B' | 1,440 | 0.56% | 26,454 | 1.84% |
| addr: 'N°' | 1 | 0.00% | 105,339 | 7.34% |
| addr: floor/apt ('etage','appt','apt','bat') | 1,850 | 0.71% | 8,038 | 0.56% |
| addr: 5-digit postcode | 1,005 | 0.39% | 7,228 | 0.50% |
| name: legal form SARL/EURL/SAS/SASU/SCI/SNC/EI/SA | 178,764 | 68.90% | 786,201 | 54.79% |
| name: dotted legal form 'S.A.R.L.'/'S.A.S.' | 0 | 0.00% | 34,245 | 2.39% |
| name: '(France)' | 20,793 | 8.01% | 92,940 | 6.48% |
| name: web-style (.com/.fr/www) | 0 | 0.00% | 49,666 | 3.46% |
| name: acronym-only (2-4 capitals, maybe + legal form) | 0 | 0.00% | 23,830 | 1.66% |
| name: 'dba'/'formerly'/'aka' | 3 | 0.00% | 7,259 | 0.51% |
