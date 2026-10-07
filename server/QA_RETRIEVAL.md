# Free-question retrieval

`app/qa_retrieval.py` replaces the fixed top-three-page lookup for free text
questions. It searches the full question and its clauses with existing BM25,
combines ranks and favors coverage of different query terms. Explicit references
such as `pages 13 à 16` take priority. Visual questions add pages 3–5 (basic
conventions). Current-KC pages are included when space remains.

Limits: 10 pages, 24,000 context characters. Each selected page receives a share
of the budget; long pages contribute query-matching passages rather than only
their beginning. Logs record requested/selected/truncated pages and the actual
source pages supplied to the model. Unknown questions do not fall back to
arbitrary pages. Explicit ranges larger than the page cap are capped.

This is lexical multi-query retrieval, not semantic embeddings or a learned
reranker. It adds no retrieval API calls, but the larger prompt can increase
answer-generation cost. Offline tests verify range handling, convention-page
coverage, bounded context and inclusion of later-page matches. Answer quality
still requires evaluation against source-grounded questions.
