# Token-to-Character Detokenization

Implementation of the beam-summing algorithm from
["From Language Models over Tokens to Language Models over Characters"](https://arxiv.org/abs/2412.03719)
(Vieira et al., 2024).

## The problem

A tokenized LM assigns probabilities to sequences of multi-character tokens
(e.g. `["Hello", ",", " world"]`). We want a character-level LM that assigns
probabilities to individual characters. The catch: a single character prefix
like `"hel"` can arise from many tokenizations (`["hel"]`, `["h","el"]`,
`["he","l"]`, ...), each with its own probability. We need to marginalize
over all of them.

## Vocabulary trie (`buildVocabTrie`)

Decomposes each token string into a character trie. For vocabulary
`["h", "he", "hel", "hello", "help", "el", "l"]`:

```
root
├─ 'h' ─── (token: "h")
│   └─ 'e' ─── (token: "he")
│       └─ 'l' ─── (token: "hel")
│           ├─ 'l'
│           │   └─ 'o' ─── (token: "hello")
│           └─ 'p' ─── (token: "help")
├─ 'e'
│   └─ 'l' ─── (token: "el")
└─ 'l' ─── (token: "l")
```

A node where `token` is set marks a complete vocabulary entry; its children
lead to longer tokens sharing that prefix.

## Mass computation (`computeLogMasses`)

Given the LM's next-token distribution, DFS the trie to compute for every
node the total probability of all tokens reachable from it:

```
mass(n) = P(n.token) + Σ_child mass(child)
```

This lets us read off character transition probabilities:

```
P(char c | node) = mass(child_c) / mass(node)
```

For example, if the LM says P("hel")=0.3, P("hello")=0.5, P("help")=0.2,
then the node after `h→e→l` has mass 1.0, and P(next char = 'l') = 0.5,
P(next char = 'p') = 0.2. The remaining 0.3 is the probability that the
token *ends* here (it's "hel", not a longer token) — this is handled by
extension (below).

## Beam candidates

Each candidate represents one possible tokenization of the character prefix
consumed so far. It carries:

| Field         | Meaning                                              |
|---------------|------------------------------------------------------|
| `tokenPrefix` | Committed tokens so far (e.g. `["he"]`)              |
| `node`        | Position in the vocab trie (partway through current token) |
| `logWeight`   | log P(this tokenization)                             |
| `snapshot`    | `LMSnapshot`: the token distribution + precomputed masses for this token prefix |

Within a single token (no commitment), advancing by a character just walks
deeper into the trie. The `snapshot` doesn't change — same LM state, same
masses, just a different node.

## Advancing the beam (`advanceBeam` — paper's `enum_cover`)

To advance the beam by character `c`:

1. **Direct trie advance**: For each candidate, try to follow `c` in the
   trie (staying within the current token). Weight update:
   `logWeight += log(mass(child_c) / mass(node))`.

2. **Extend**: For candidates sitting on a completed token, commit it —
   append the token to `tokenPrefix`, return to the trie root, take a fresh
   `LMSnapshot` (new LM call + mass computation), then follow `c` from root.
   Weight update for the commit: `logWeight += log(P(token) / mass(node))`.
   This is the fraction of mass at this node belonging to the specific
   completed token (vs. longer tokens passing through).

3. **Prune** (paper's `prune_top_K_buckets`): Keep only the top K
   candidates whose relative probability exceeds the threshold.

Extended candidates are pruned *before* materialization to avoid unnecessary
LM calls.

## Next-character probabilities (`nextCharProbs` — paper's `next_char_probability`)

Marginalize over the beam to produce a character-level distribution:

For each candidate and each character child `c` of its trie node:
```
contribution(c) = exp(logWeight) × mass(child_c) / mass(node)
```

Candidates on completed tokens are also extended: the committed-token
probability flows through a fresh LM distribution from the trie root,
contributing to all first-characters of the new token.

Sum contributions across all candidates per character, then normalize.

### Why extension matters for `nextCharProbs`

Consider a candidate at the node for "the" (a complete token) that also has
children leading to "them", "there", etc.:

- **Without extending**: contributions come from longer tokens ("them",
  "there", ...) — probability that the actual token is longer.
- **After extending**: commit "the", get a fresh LM distribution, and
  contribute from the root — probability that the token is exactly "the"
  and the *next* token starts with various characters.

Both paths are needed for correct character probabilities.

## Caching in `detokenize`

The public `detokenize` function wraps everything with two caches:

- **`snapshotCache`** (keyed by token prefix): Avoids redundant LM calls
  when multiple candidates share a token prefix. Also avoids recomputing
  masses.

- **`beamCache`** (keyed by character prefix): Incrementally builds beam
  states, so querying prefix `[a,b,c]` reuses the beam already computed
  for `[a,b]`.

## Paper correspondence

| Paper algorithm           | Our code                                      |
|---------------------------|-----------------------------------------------|
| `enum_cover`              | `tryExtend` + materialization in `advanceBeam` |
| `prune_top_K_buckets`     | `pruneBeam`                                   |
| `next_char_probability`   | `nextCharProbs`                               |
| Token trie                | `buildVocabTrie` + `VocabTrieNode`            |
| Weight propagation        | `computeLogMasses` (DFS instead of sparse matrices) |

## Simplifications vs. the reference implementation ([genlm-bytes](https://github.com/genlm/genlm-bytes))

- No byte-level handling (we work at character level, ignoring partial UTF-8).
- No sparse matrix acceleration for mass computation (simple DFS suffices
  for our vocabulary sizes).
- No adaptive token healing (when the beam empties, we just return nothing).
- No EOS (end-of-sequence) token handling.
- Stateless `LanguageModel<T>` interface instead of stateful KV-cache
  management — caching handles the performance concern.
