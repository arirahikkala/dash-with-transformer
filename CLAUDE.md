A modernized Dasher, an information-theory-based input method.

Dasher implements a mapping between a language model (as the set of all strings that can be sampled from it, along with
their probabilities) and a unit square. The strings are organized by length from left to right and alphabetically from
top to bottom. The intent is for users to be able to input strings by "gliding" rightward into the language model,
choosing the sentence's content by going up or downward, and correcting mistakes by going leftward.

Currently implement as a Vite TypeScript frontend-only project. The language model is a character Markov trigram model (`model.bin`), planned to be swapped for a proper LLM later.

There are basically two hard things that the front-end does:

- Scene construction (scene.ts): Find the content (tree of prefixes to display) in a window centered on a given cursor position
- Cursor normalization (cursor.ts): Find the canonical form of a cursor

... and a number of easier things, like rendering a constructed scene, animating the widget, etc.

## Geometric model

To describe the mapping more precisely:

An autoregressive language model induces a recursive tiling of the unit square [0,1]×[0,1]. Every text prefix p maps to a _square_:

    width = height = P(p)          (joint probability of the prefix)
    right edge at x = 1
    left  edge at x = 1 − P(p)
    y-position determined by cumulative conditional probabilities

Inside each square, the next-token distribution carves out child squares stacked vertically (in the order the model returns them). A child for token c with conditional probability p*c occupies, in the \_parent's* normalised [0,1]×[0,1] coordinate frame:

    x ∈ [1 − p_c, 1]
    y ∈ [cumBefore, cumBefore + p_c]

Because every child is narrower than its parent (unless p_c = 1), there is a "gap" on the left side of every square that no child covers.

## Cursor

Conceptually, a cursor just points at an exact (x, y) position in the unit square, a "global position".

In the actual implementation, it's a triplet (prefix, x, y), where x and y are in the current square's frame: x = 0 is the left edge, y = 0 is the top edge.

normalizeCursor in cursor.ts implements smooth navigation through different prefixes by returning a cursor at the same global position, whose prefix is the smallest square that contains the cursor. Note that this requires exact rational arithmetic (via `rational.ts`) and a recursive ascent/descent search. But what it allows is that smooth movement in the widget can be animated by simply updating (x, y) and then calling normalizeCursor.

The cursor's position outside of these calculations is kept in machine floats, so memory leaks from overly precise calculation results sticking around aren't a concern.

## Scene builder

The visible window is a square centered on the cursor with its right edge at x=1 (this is sufficient to fully describe
the displayed area). buildScene in scene.ts finds the tree of SceneNodeS within the window. Since technically
arbitrarily distantly related nodes might be visible within the window, it has to do recursive ascent/descent as well.

## Architecture

- `src/types.ts` - central types
- `src/rational.ts` — exact BigInt rational arithmetic
- `src/cursor.ts` — cursor type, normalization (ascent/descent)
- `src/scene.ts` — `buildScene`: computes the content of the visible window
- `src/render.ts` — `renderScene`: renders said content with a canvas renderer
- `src/trigram.ts` — trigram model loader (reads `model.bin`)
- `src/main.ts` — wires things together, displays widget, mouse-driven animation loop
