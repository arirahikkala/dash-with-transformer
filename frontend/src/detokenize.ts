import {
  adaptModel,
  type LanguageModel,
  type PlainLanguageModel,
  type PlainTokenProb,
} from "./types";

// =====================================================================
// Log-space arithmetic
// =====================================================================

function logSumExp(xs: readonly number[]): number {
  if (xs.length === 0) return -Infinity;
  let max = -Infinity;
  for (const x of xs) if (x > max) max = x;
  if (max === -Infinity) return -Infinity;
  let sum = 0;
  for (const x of xs) sum += Math.exp(x - max);
  return max + Math.log(sum);
}

function logAddExp(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const max = a > b ? a : b;
  return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
}

// =====================================================================
// Token-to-character detokenization
// =====================================================================
//
// Implements the beam-summing algorithm from
// "From Language Models over Tokens to Language Models over Characters"
// (Vieira et al., 2024 — https://arxiv.org/abs/2412.03719).
//
// A tokenized LM assigns probabilities to sequences of multi-character
// tokens.  To get character-level probabilities we maintain a beam of
// candidate tokenizations of the character prefix consumed so far and
// marginalize over them.
//
// Paper correspondence:
//   enum_cover           → beam extension  (tryExtend + materialize)
//   prune_top_K_buckets  → pruneBeam
//   next_char_probability → nextCharProbs

// --- Token vocabulary trie ------------------------------------------------
//
// Decomposes each vocabulary token into its characters.  A node whose
// `token` field is set marks the end of a vocabulary entry; its children
// (if any) continue to longer tokens that share the same prefix.

interface VocabTrieNode {
  readonly children: Map<number, VocabTrieNode>; // char code → child
  token: string | undefined; // vocabulary entry ending here
}

function vocabTrieNode(): VocabTrieNode {
  return { children: new Map(), token: undefined };
}

function buildVocabTrie(vocab: readonly string[]): VocabTrieNode {
  const root = vocabTrieNode();
  for (const tok of vocab) {
    if (tok.length === 0) continue;
    let node = root;
    for (let i = 0; i < tok.length; i++) {
      const ch = tok.charCodeAt(i);
      let child = node.children.get(ch);
      if (!child) {
        child = vocabTrieNode();
        node.children.set(ch, child);
      }
      node = child;
    }
    node.token = tok; // last-one-wins if duplicates exist
  }
  return root;
}

// --- Mass computation -----------------------------------------------------
//
// Given the LM's next-token log-probabilities, compute for every trie
// node the total probability of all vocabulary tokens reachable from it:
//
//     mass(n) = P(n.token) + Σ_child mass(child)
//
// This lets us read off character transition probabilities:
//     P(char c | node) = mass(child_c) / mass(node)

function computeLogMasses(
  root: VocabTrieNode,
  logProbs: ReadonlyMap<string, number>,
): Map<VocabTrieNode, number> {
  const result = new Map<VocabTrieNode, number>();

  function dfs(node: VocabTrieNode): number {
    const parts: number[] = [];
    if (node.token !== undefined) {
      const lp = logProbs.get(node.token);
      if (lp !== undefined && lp > -Infinity) parts.push(lp);
    }
    for (const child of node.children.values()) {
      const cm = dfs(child);
      if (cm > -Infinity) parts.push(cm);
    }
    const m = logSumExp(parts);
    result.set(node, m);
    return m;
  }

  dfs(root);
  return result;
}

// --- LM info (cached per token prefix) ------------------------------------

interface LMSnapshot {
  readonly logTokenProbs: ReadonlyMap<string, number>;
  readonly logMasses: Map<VocabTrieNode, number>;
}

async function takeLMSnapshot(
  model: PlainLanguageModel<readonly string[], string>,
  trieRoot: VocabTrieNode,
  tokenPrefix: readonly string[],
): Promise<LMSnapshot> {
  const dist = await model(tokenPrefix);
  const logTokenProbs = new Map<string, number>();
  for (const { token, probability } of dist) {
    if (probability > 0) logTokenProbs.set(token, Math.log(probability));
  }
  return {
    logTokenProbs,
    logMasses: computeLogMasses(trieRoot, logTokenProbs),
  };
}

// --- Beam candidates ------------------------------------------------------
//
// Each candidate represents one tokenization of the character prefix
// consumed so far.  It sits at some node in the vocabulary trie (partway
// through the current token) and carries the LM's probability snapshot
// for its committed token sequence.

interface Candidate {
  readonly tokenPrefix: readonly string[]; // committed tokens
  readonly node: VocabTrieNode; // position in current partial token
  readonly logWeight: number; // log P(this tokenization)
  readonly snapshot: LMSnapshot; // LM state at this token prefix
}

/** Advance one character within the trie (no token commitment). */
function advanceCandidate(c: Candidate, ch: number): Candidate | undefined {
  const child = c.node.children.get(ch);
  if (!child) return undefined;
  const massCur = c.snapshot.logMasses.get(c.node) ?? -Infinity;
  const massChild = c.snapshot.logMasses.get(child) ?? -Infinity;
  if (massChild === -Infinity) return undefined;
  return {
    tokenPrefix: c.tokenPrefix,
    node: child,
    logWeight: c.logWeight + massChild - massCur,
    snapshot: c.snapshot, // same LM state, same masses
  };
}

/**
 * If the candidate sits on a completed token, return the token prefix
 * and weight for committing it (returning to the trie root).  The
 * result is "unmaterialized" — it still needs an LM call to get the
 * new snapshot.
 */
function tryExtend(
  c: Candidate,
): { tokenPrefix: readonly string[]; logWeight: number } | undefined {
  if (c.node.token === undefined) return undefined;
  const logP = c.snapshot.logTokenProbs.get(c.node.token);
  if (logP === undefined || logP === -Infinity) return undefined;
  const massCur = c.snapshot.logMasses.get(c.node) ?? -Infinity;
  if (massCur === -Infinity) return undefined;
  // Weight update: P(this specific token) / P(any token through this node)
  return {
    tokenPrefix: [...c.tokenPrefix, c.node.token],
    logWeight: c.logWeight + logP - massCur,
  };
}

// --- Beam operations (paper: enum_cover + prune_top_K_buckets) -----------

interface Beam {
  readonly candidates: readonly Candidate[];
}

/** Prune to top K candidates above the threshold (prune_top_K_buckets). */
function pruneBeam(beam: Beam, K: number, logPruneThreshold: number): Beam {
  const logZ = logSumExp(beam.candidates.map((c) => c.logWeight));
  const kept = beam.candidates
    .filter((c) => c.logWeight - logZ > logPruneThreshold)
    .sort((a, b) => b.logWeight - a.logWeight)
    .slice(0, K);
  return { candidates: kept };
}

/**
 * Advance the entire beam by one character (enum_cover for one step).
 *
 * For each candidate:
 *   1. Try to follow the character directly in the trie.
 *   2. If the candidate can commit a token (extend), do so, materialize
 *      the new LM state at the trie root, then follow the character.
 */
async function advanceBeam(
  beam: Beam,
  ch: number,
  trieRoot: VocabTrieNode,
  getSnapshot: (tokenPrefix: readonly string[]) => Promise<LMSnapshot>,
  K: number,
  logPruneThreshold: number,
): Promise<Beam> {
  const result: Candidate[] = [];

  // Direct trie advances
  for (const c of beam.candidates) {
    const adv = advanceCandidate(c, ch);
    if (adv) result.push(adv);
  }

  // Extend (commit token → root), prune, materialize, then advance
  let logZ = logSumExp(result.map((c) => c.logWeight));
  const exts: { tokenPrefix: readonly string[]; logWeight: number }[] = [];
  for (const c of beam.candidates) {
    const ext = tryExtend(c);
    if (ext) {
      logZ = logAddExp(logZ, ext.logWeight);
      exts.push(ext);
    }
  }

  const materialized = await Promise.all(
    exts
      .filter((e) => e.logWeight - logZ > logPruneThreshold)
      .map(
        async (e): Promise<Candidate> => ({
          tokenPrefix: e.tokenPrefix,
          node: trieRoot,
          logWeight: e.logWeight,
          snapshot: await getSnapshot(e.tokenPrefix),
        }),
      ),
  );

  for (const c of materialized) {
    const adv = advanceCandidate(c, ch);
    if (adv) result.push(adv);
  }

  return pruneBeam({ candidates: result }, K, logPruneThreshold);
}

// --- Next-character probabilities (paper: next_char_probability) ----------

/**
 * Marginalize over the beam to produce a character-level distribution.
 *
 * For each candidate, character c gets probability mass:
 *     weight × mass(child_c) / mass(node)
 *
 * Candidates that sit on a completed token are also extended: the
 * committed-token probability flows through a fresh LM distribution
 * from the trie root.
 */
async function nextCharProbs(
  beam: Beam,
  trieRoot: VocabTrieNode,
  getSnapshot: (tokenPrefix: readonly string[]) => Promise<LMSnapshot>,
): Promise<readonly PlainTokenProb<number>[]> {
  // Accumulate per-character log contributions from every candidate.
  const contribs = new Map<number, number[]>();

  function addContributions(c: Candidate) {
    const massCur = c.snapshot.logMasses.get(c.node) ?? -Infinity;
    if (massCur === -Infinity) return;
    for (const [ch, child] of c.node.children) {
      const massChild = c.snapshot.logMasses.get(child) ?? -Infinity;
      if (massChild === -Infinity) continue;
      let arr = contribs.get(ch);
      if (!arr) {
        arr = [];
        contribs.set(ch, arr);
      }
      arr.push(c.logWeight + massChild - massCur);
    }
  }

  // Contributions from candidates as-is (staying within current token)
  for (const c of beam.candidates) addContributions(c);

  // Contributions from extended candidates (commit token, start new one)
  const extPromises: Promise<Candidate>[] = [];
  for (const c of beam.candidates) {
    const ext = tryExtend(c);
    if (ext) {
      extPromises.push(
        getSnapshot(ext.tokenPrefix).then(
          (snapshot): Candidate => ({
            tokenPrefix: ext.tokenPrefix,
            node: trieRoot,
            logWeight: ext.logWeight,
            snapshot,
          }),
        ),
      );
    }
  }
  for (const c of await Promise.all(extPromises)) addContributions(c);

  // Combine via logsumexp and normalize.
  const entries: { ch: number; logP: number }[] = [];
  for (const [ch, parts] of contribs) {
    entries.push({ ch, logP: logSumExp(parts) });
  }
  const totalLogP = logSumExp(entries.map((e) => e.logP));
  if (totalLogP === -Infinity) return [];

  return entries
    .map((e) => ({ token: e.ch, probability: Math.exp(e.logP - totalLogP) }))
    .sort((a, b) => a.token - b.token);
}

// --- Public API -----------------------------------------------------------

/**
 * Convert a token-level LanguageModel into a character-level
 * LanguageModel (where tokens are char codes).
 *
 * @param tokenModel  The underlying token-level model.
 * @param vocab       Complete token vocabulary (each entry is the string
 *                    a token decodes to).
 * @param K           Beam width — max candidate tokenizations to track.
 * @param pruneThreshold  Drop candidates whose relative probability falls
 *                        below this (0 = never prune by threshold).
 */
export function detokenize(
  tokenModel: PlainLanguageModel<readonly string[], string>,
  vocab: readonly string[],
  K: number,
  pruneThreshold = 0,
): LanguageModel<readonly number[], number> {
  const trie = buildVocabTrie(vocab);
  const logPruneThreshold =
    pruneThreshold > 0 ? Math.log(pruneThreshold) : -Infinity;

  // Cache LM snapshots by token prefix (avoids redundant LM calls
  // and mass recomputation when multiple candidates share a prefix).
  const snapshotCache = new Map<string, Promise<LMSnapshot>>();
  function getSnapshot(tokenPrefix: readonly string[]): Promise<LMSnapshot> {
    const key = JSON.stringify(tokenPrefix);
    let cached = snapshotCache.get(key);
    if (!cached) {
      cached = takeLMSnapshot(tokenModel, trie, tokenPrefix);
      snapshotCache.set(key, cached);
    }
    return cached;
  }

  // Cache beam states by character prefix.  Each entry is built
  // incrementally from its parent (one character shorter).
  const beamCache = new Map<string, Promise<Beam>>();
  function getBeam(prefix: readonly number[]): Promise<Beam> {
    const key = prefix.join(",");
    let cached = beamCache.get(key);
    if (!cached) {
      cached = (async (): Promise<Beam> => {
        if (prefix.length === 0) {
          return {
            candidates: [
              {
                tokenPrefix: [] as string[],
                node: trie,
                logWeight: 0,
                snapshot: await getSnapshot([]),
              },
            ],
          };
        }
        const parent = await getBeam(prefix.slice(0, -1));
        return advanceBeam(
          parent,
          prefix[prefix.length - 1],
          trie,
          getSnapshot,
          K,
          logPruneThreshold,
        );
      })();
      beamCache.set(key, cached);
    }
    return cached;
  }

  return adaptModel(async (prefix: readonly number[]) => {
    const beam = await getBeam(prefix);
    return nextCharProbs(beam, trie, getSnapshot);
  });
}
