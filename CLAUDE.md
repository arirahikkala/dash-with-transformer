The repo is split into `frontend/` (Vite + TypeScript) and `backend/` (Python + FastAPI).

## Concepts and central types

### Geometric model

An autoregressive language model induces a recursive, tree-structured tiling of the unit square [0,1]×[0,1]. Every text prefix p maps to a _square_:

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

`normalizeCursor` in `cursor.ts` implements smooth navigation through different prefixes by returning a cursor at the same global position, whose prefix is the smallest square that contains the cursor. This uses a recursive ascent/descent search. Smooth movement in the widget can be animated by simply updating (x, y) and then calling normalizeCursor.

### LanguageModel

A view into the cumulative distribution function of a language model's next-token distribution at a given prefix. Tokens are returned with their *extents* in the CDF.

## Backend

A Python FastAPI server (`backend/`) that adapts a tokenized language model into a byte-level one via `genlm-bytes`.

## Testing

Run `npm test` (or `npx vitest run`) in `frontend/` to execute the test suite. Tests use Vitest and live alongside source files as `*.test.ts`.

## Architecture

### Frontend (`frontend/src/`)

- `types.ts` — central types
- `cursor.ts` — cursor type, `normalizeCursor` (ascent/descent)
- `scene.ts` — `buildScene`: computes the content of the visible window
- `render.ts` — `renderScene`: renders said content with a canvas renderer
- `main.ts` — wires things together, displays widget, mouse-driven animation loop
- `models.ts` — model-to-model adapters, most notably `fromByteLevelModel` which converts UTF-8 byte models to Unicode codepoint models
- `lstm`, `webgpu`, `remote` — subdirectories for the (front-end) LSTM and WebGPU, and (backend) remote inference modes

### Library-ish front-end code

- `async-iterables`.ts — utilities for racing async iterables

### Backend (`backend/dash_with_transformer/`)

- `server.py` — FastAPI app, single `/predict` endpoint
- `engine.py` — `BytePredictionEngine`: beam-search inference, LRU state cache, trie expansion
