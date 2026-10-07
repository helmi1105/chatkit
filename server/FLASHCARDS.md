# Flashcards with optional SVG

The compact card groups topic, optional SVG, short question and reveal button.
After reveal it shows the answer, source and three ratings. `A revoir` requeues
after one intervening card when available, `Difficile` requeues at the end,
and `Je sais` does not requeue. Each rating advances the same widget. This is a
within-session review queue, not time-based spaced repetition. Its position,
rating history and latest self-ratings persist in the learner session.
The counter identifies the card within the original deck; remaining passes
include repeated cards. Old reveal/review/next buttons remain supported.
Ratings produce `flashcard_rated` events with `evidence_role=self_report_only`;
they never validate mastery. Repeated stale button actions do not add ratings.

New decks can include SVG on the front (identify a symbol) or back (recall its
appearance). A native Image widget displays the validated SVG directly in chat
using an embedded PNG data URL rendered from the validated SVG; no diagram button
or side viewer is needed. PNGs are cached in app/generated_flashcards under
content hashes. The widget carries the bytes directly, avoiding HTTPS iframe
requests to HTTP loopback addresses. Existing messages retain their old URLs;
reopen flashcards to render the current deck again without regenerating it.
Back-side drawings are sent only after reveal. Front-side drawings omit source
links until reveal; generation is instructed to omit answer labels. This is not
an automatic semantic check for answer leakage or doctrinal correctness.
Invalid front drawings cause the card to be dropped; invalid back drawings fall
back to the textual answer. Legacy text decks remain usable; finish an existing
deck and reopen flashcards to generate a new one with optional drawings.
Flashcards retain their existing selected text provider/model. SVGs are generated
with the card text, without an additional illustration call.

During learning, click Cartes de r?vision or type `flashcards` (`fiches` also works).
A deck contains 3?5 text questions for the current KC. Voir la r?ponse reveals the
explanation, source page and PDF link. ? revoir records a self-report; Carte
suivante advances. Reopening flashcards resumes the deck; reopening a completed
deck generates a new one. The deck and review markers persist in the session.

Cards use only the current KC page excerpts. Invalid pages, blank fields and
duplicate questions are rejected/filtered. This checks structural/source-page
validity, not semantic correctness: generated explanations are not expert-approved.
No PDF symbol crops are generated; SVG drawings are schematic illustrations.

One model call is charged per new deck. Reveal/review/next require no model calls.
Cards are unavailable during pending quizzes and initial diagnostic screening.
Viewing, revealing or marking cards never changes mastery, validation or diagnostic
evidence. Events: flashcards_generated, flashcard_revealed,
flashcard_marked_for_review. These are learning/self-report events, not assessments.

The existing automated learner simulation does not choose flashcards automatically.
Offline tests: python -m unittest test_flashcards -q (from server).
