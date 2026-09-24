"""FROZEN evaluation harness. Only the lead changes files in this package.

Any change alters `eval_code_sha256`; results produced with a hash different from
planning/eval_lock.json are flagged EVAL_MODIFIED and ignored by the planner.
"""
