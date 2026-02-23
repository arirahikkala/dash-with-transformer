A modernized Dasher, an information-theory-based input method.

Dasher implements a mapping between a language model (as the set of all strings that can be sampled from it, along with
their probabilities) and a unit square. The strings are organized by length from left to right and alphabetically from
top to bottom. The intent is for users to be able to input strings by "gliding" rightward into the language model,
choosing the sentence's content by going up or downward, and correcting mistakes by going leftward.

The repo is split into `frontend/` (Vite + TypeScript) and `backend/` (Python + FastAPI).

## Geometric model

To describe the mapping more precisely:

An autoregressive language model induces a recursive tiling of the unit square [0,1]×[0,1]. Every text prefix p maps to a _square_:

    width = height = P(p)          (joint probability of the prefix)
    right edge at x = 1
    left  edge at x = 1 − P(p)
    y-position determined by cumulative conditional probabilities

Inside each square, the next-token distribution carves out child squares stacked vertically. A child for token c with conditional probability p*c occupies, in the \_parent's* normalised [0,1]×[0,1] coordinate frame:

    x ∈ [1 − p_c, 1]
    y ∈ [cumBefore, cumBefore + p_c]

Because every child is narrower than its parent (unless p_c = 1), there is a "gap" on the left side of every square that no child covers.

## Cursor

Conceptually, a cursor just points at an exact (x, y) position in the unit square, a "global position".

In the actual implementation, it's a triplet (prefix, x, y), where x and y are in the current square's frame: x = 0 is the left edge, y = 0 is the top edge.

`normalizeCursor` in `cursor.ts` implements smooth navigation through different prefixes by returning a cursor at the same global position, whose prefix is the smallest square that contains the cursor. Note that this requires exact rational arithmetic (via `rational.ts`) and a recursive ascent/descent search. But what it allows is that smooth movement in the widget can be animated by simply updating (x, y) and then calling normalizeCursor.

The cursor's position outside of these calculations is kept in machine floats, so memory leaks from overly precise calculation results sticking around aren't a concern.

## Scene builder

The visible window is a square centered on the cursor with its right edge at x=1 (this is sufficient to fully describe
the displayed area). buildScene in scene.ts finds the tree of SceneNodeS within the window. Since technically
arbitrarily distantly related nodes might be visible within the window, it has to do recursive ascent/descent as well.

## LanguageModel

`LanguageModel<P, T>` in `types.ts` is the central abstraction for querying next-token distributions. Its signature:

    (prefix, rangeStart, rangeEnd, minSize, specificToken?) → AsyncIterable<TokenProb<T>>

Each returned `TokenProb` has a `start` and `end` on the cumulative probability line [0, 1]. These extents depend only on the prefix — the query parameters just control which entries are returned. Entries may be returned in any order.

- **rangeStart / rangeEnd** — closed range `[rangeStart, rangeEnd]`. Only entries whose extent touches this range are returned. A point query (`start === end`) returns the 1–2 entries at that exact point.
- **minSize** — only entries with probability ≥ minSize are returned.
- **specificToken** — if set, return only that token's extent (ignoring range/size), doing minimal work. Used by the ascent paths in cursor.ts and scene.ts to look up a known token without materializing the full distribution.

Two implementations exist:

- `adaptModel` (types.ts) — wraps a simple `prefix → {token, probability}[]` function, computing cumulative extents and applying filters. Currently used only in tests.
- `fromByteLevelModel` (models.ts) — adapts a byte-level UTF-8 model into a codepoint-level model, using exact rational arithmetic for cumulative positions and recursive expansion of multi-byte sequences. This is the main production implementation.

**Performance note:** the type is generic, but the main use at this time is representing a Unicode codepoint model via `fromByteLevelModel`. A query with `minSize=0`, full range, and no `specificToken` materializes the entire distribution over all Unicode codepoints present in the model — expanding every multi-byte group one byte-level query at a time. Hence, _every LanguageModel call must have a nonzero minSize or specificToken_ set.

## Backend

A Python FastAPI server (`backend/`) that wraps a byte-level language model via `genlm-bytes`.

## Testing

Run `npm test` (or `npx vitest run`) in `frontend/` to execute the test suite. Tests use Vitest and live alongside source files as `*.test.ts`.

## Architecture

### Frontend (`frontend/src/`)

- `types.ts` — central types
- `rational.ts` — exact BigInt rational arithmetic
- `cursor.ts` — cursor type, normalization (ascent/descent)
- `scene.ts` — `buildScene`: computes the content of the visible window
- `render.ts` — `renderScene`: renders said content with a canvas renderer
- `main.ts` — wires things together, displays widget, mouse-driven animation loop
- `models.ts` — upconversion from byte-level model to Unicode codepoint LanguageModel
- `backend.ts` — `predictBytes`: backend client, request batching and trie cache

### Backend (`backend/llm_dasher/`)

- `server.py` — FastAPI app, single `/predict` endpoint
- `engine.py` — `BytePredictionEngine`: beam-search inference, LRU state cache, trie expansion

### Overall

Hence the overall architecture is basically a straight pull pipeline:

renderScene <- buildScene <- fromByteLevelModel <- predictBytes <- (HTTP req) <- predict_batch