# YojanaMitra frontend

## Build
- Replace the starter screen with a polished, responsive YojanaMitra experience using the requested dark teal/forest palette, warm amber accents, large readable type, and restrained cards.
- Create the input screen with plain-language situation field, language selector, visual-only microphone control, and submit action.
- Add an in-memory flow for loading, an optional clarifying question with answer/skip actions, and results.
- Keep all mock access behind one `fetchMatches(query, language)` function so it can later be replaced by a real request.

## Results experience
- Show a compact interpretation summary with every profile field and its confidence.
- Render three varied mock scheme examples so Eligible, Needs checking, and Not eligible are visible together.
- Include each scheme’s level, reason, conditions still requiring confirmation, expandable exact eligibility text, documents, and external Apply link.
- Keep the supplied JSON shape unchanged while adding the third example in the same structure.

## Quality
- Define the visual system centrally with semantic color and typography tokens.
- Add accessible labels, focus states, clear disabled/loading behavior, keyboard-friendly expandable rules, and mobile-friendly layouts.
- Add page-specific title and social metadata, then verify the finished screen in desktop and mobile-sized browser views.
