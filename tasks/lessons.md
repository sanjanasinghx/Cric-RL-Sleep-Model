# Lessons

Review this file at the start of each session.

## L1. Verify the identity of any file you are told is the base before using it.
Pattern: the brief named `~/Downloads/README.md` as the corrected base. It was byte-for-byte the
original uncorrected README. The real corrected file was `~/Downloads/README_2.md`.
Rule: diff a supplied base file against the thing it claims to replace before acting on it. If it
contradicts its description, stop and surface it rather than proceeding.

## L2. GitHub renders README math with markdown emphasis parsing running before KaTeX.
Pattern: a bare `*` or `_` inside `$...$` is consumed as an italic marker, so `s_2^*` reaches KaTeX
as a dangling superscript and fails with "Double subscripts: use braces to clarify".
Rule: brace every subscript and superscript even single characters, write `\ast` not a literal star,
and use `\mathbf{1}` not `\mathbb{1}`. Validate with a local KaTeX render plus a strip-all-`*`/`_`
pass that simulates the markdown step.

## L3. Do not commit a reference to a file that is not committed.
Pattern: Phase 1 was tempted to add a README line pointing to `artifacts/metrics.json` while the file
itself was deferred to Phase 3.
Rule: the pointer line and the file it points to land in the same commit.
