# SVG in free-question answers

Free text questions can return an explanation and an optional SVG drawing.
Ask for a drawing in the chat; when supported by the retrieved course excerpts,
the answer includes a `Voir le schema` button opening the existing viewer.
Quiz, lesson and uploaded-image response generation are unchanged.

`app/svg_answer.py` defines the structured output and validates a restricted SVG
subset. Scripts, event handlers, CSS, external resources and XML declarations
are rejected. Source pages must belong to the supplied excerpts. Invalid SVG
is omitted while preserving the explanation. Drawings are labelled schematic;
validation checks markup and source-page membership, not doctrinal correctness.

No extra model call is made for the SVG: it accompanies the free-answer output.
The current text provider generates it. No image-generation API is used.

Offline tests (from server): `python -m unittest test_svg_answer`
