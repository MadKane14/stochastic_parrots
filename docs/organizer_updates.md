# Organizer updates (verbatim, newest first)

## Sat 26 Sep — candidate_pairs.tsv is part of the final submission
> Blocking has to scale. Amazon resolves business entities across billions of records, so comparing every record with
> every other one is not an option. Your blocking / candidate-generation step must cut the search space to a small
> candidate set per Source 1 entity.
> Candidate generation counts toward the final ranking. We will review your candidate_pairs.tsv and the code that
> produces it when deciding final rankings, alongside your matching_results.tsv score. The approach that generates a
> smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private
> leaderboard.

Implications for us (lead): report cands/S1 next to every F0.5; prefer changes that keep recall while shrinking the
set; the README defines candidate_pairs.tsv as the LAST filtering stage before the final model, so a cheap
learned pruning stage (blocker → light scorer keeps top-few per S1) legitimately shrinks the reported set.
