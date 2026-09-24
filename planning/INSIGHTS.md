# INSIGHTS — append-only log (planner drafts, lead commits)

Format: `I-NN [confidence] statement — evidence: run_ids / exploration script — (round, date)`
Superseded insights are struck through with a pointer, never deleted.

## Prior knowledge (from the problem statement — not yet verified on data)
- I-01 [given] Metric is macro F0.5 per S1 entity incl. singletons; a single false match on a singleton costs a full 1.0 on that entity → the decision rule must be conservative, and the singleton rate sets how conservative. — problem statement
- I-02 [given] S1 is deduplicated → an S2/S3 record should belong to at most one S1 entity (enables "exclusive right" assignment). To verify in EDA (T03 Q3). — problem statement
- I-03 [given] Test contains France, unseen in train → components must generalize across countries; `xc_*` splits are our proxy. — problem statement
- I-04 [given] candidate_pairs.tsv is audited for blocking quality (recall ceiling, reduction ratio) → keep candidate sets reasonably tight, not "everything". — problem statement
