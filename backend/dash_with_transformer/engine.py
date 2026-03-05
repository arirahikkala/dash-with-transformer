from __future__ import annotations

import asyncio
import os
from collections import OrderedDict

import numpy as np
from genlm.backend import load_model_by_name, decode_vocab
from genlm.bytes import BeamParams, ByteBeamState
from genlm.bytes.byte_lm.trie_state import TrieMode

BEAM_K = int(os.environ.get("BEAM_K", "5"))
PRUNE_THRESHOLD = float(os.environ.get("PRUNE_THRESHOLD", "0.05"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2-medium")
MAX_MODEL_LEN = int(os.environ["MAX_MODEL_LEN"]) if "MAX_MODEL_LEN" in os.environ else None
CACHE_MAX_SIZE = int(os.environ.get("CACHE_MAX_SIZE", "10000"))
INITIAL_CONTEXT = (
    [int(x) for x in os.environ["INITIAL_CONTEXT"].split(",")]
    if "INITIAL_CONTEXT" in os.environ
    else None
)
SPECIAL_TOKENS: list[int] = (
    [int(x) for x in os.environ["SPECIAL_TOKENS"].split(",")]
    if os.environ.get("SPECIAL_TOKENS")
    else []
)


class BytePredictionEngine:
    def __init__(self):
        self._cache: OrderedDict[tuple[int, ...], ByteBeamState] = OrderedDict()
        self._lock = asyncio.Lock()
        self._special_token_bytes: list[bytes] = []
        self._special_token_labels: list[str] = []

    async def start(self):
        engine_opts = {}
        if MAX_MODEL_LEN is not None:
            engine_opts["max_model_len"] = MAX_MODEL_LEN
        llm = load_model_by_name(MODEL_NAME, backend="vllm",
            llm_opts={"engine_opts": engine_opts})

        # Validate and resolve special tokens
        if len(SPECIAL_TOKENS) != len(set(SPECIAL_TOKENS)):
            raise ValueError(f"SPECIAL_TOKENS contains duplicates: {SPECIAL_TOKENS}")
        all_special_ids = set(llm.tokenizer.all_special_ids)
        byte_vocab, str_vocab = decode_vocab(llm.tokenizer)
        self._special_token_bytes = [byte_vocab[tid] for tid in SPECIAL_TOKENS]
        self._special_token_labels = [str_vocab[tid] for tid in SPECIAL_TOKENS]

        params = BeamParams(
            K=BEAM_K,
            prune_threshold=PRUNE_THRESHOLD,
            **({"special_tokens": self._special_token_bytes} if self._special_token_bytes else {}),
        )
        initial = await ByteBeamState.initial(llm, params,
            **({"initial_context": INITIAL_CONTEXT} if INITIAL_CONTEXT is not None else {}))
        # Store the initial state in WITHOUT_EOS mode for prefix caching.
        # prefill() internally operates in WITHOUT_EOS, so all cached states
        # use this mode. We switch to WITH_EOS only at query time.
        self._cache[()] = initial.with_mode(TrieMode.WITHOUT_EOS)
        self._llm = llm
        self._initial = initial

    async def shutdown(self):
        await self._initial.cleanup()

    def _longest_cached_prefix(self, ctx: tuple[int, ...]) -> tuple[tuple[int, ...], ByteBeamState]:
        """Find the longest prefix of ctx that exists in the cache."""
        for i in range(len(ctx), -1, -1):
            prefix = ctx[:i]
            if prefix in self._cache:
                # Move to end for LRU
                self._cache.move_to_end(prefix)
                return prefix, self._cache[prefix]
        # Should never happen since () is always in cache
        raise RuntimeError("empty prefix missing from cache")

    def _put_cache(self, key: tuple[int, ...], state: ByteBeamState):
        """Insert into the cache with LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        self._cache[key] = state
        while len(self._cache) > CACHE_MAX_SIZE:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            if evicted_key == ():
                # Never evict the empty-prefix sentinel; re-insert at front and stop.
                self._cache[evicted_key] = evicted_val
                self._cache.move_to_end(evicted_key, last=False)
                break

    async def _predict_one(self, ctx: tuple[int, ...]) -> list[float]:
        """Get next-token probabilities for a context of byte/special-token indices."""
        async with self._lock:
            prefix, state = self._longest_cached_prefix(ctx)

        # Advance one element at a time through the uncached suffix
        suffix = ctx[len(prefix):]
        for i, val in enumerate(suffix):
            # In our indexing, 0-255 are bytes, 256+ are special tokens.
            # genlm-bytes uses 256-257 internally, so special tokens start at 258.
            genlm_val = val if val < 256 else val + 2
            state = await (state.prune() << genlm_val)
            new_prefix = ctx[: len(prefix) + i + 1]
            async with self._lock:
                self._put_cache(new_prefix, state)

        # Switch to WITH_EOS for generation and get log-probs
        gen_state = state.with_mode(TrieMode.WITH_EOS)
        logp = await gen_state.logp_next()
        # Indices 0-255 are bytes, 256-257 are internal to genlm-bytes, 258+ are special tokens.
        # Strip 256-257 so the returned array is [byte0..byte255, special0, special1, ...].
        probs = np.exp(np.concatenate([logp.ps[:256], logp.ps[258:]])).tolist()
        return probs

    async def _predict_trie(
        self,
        prefix: tuple[int, ...],
        prob_so_far: float,
        min_prob: float,
        depth: int = 0,
    ) -> dict:
        """Recursively build a trie of next-token distributions.

        Only expands children whose probability >= min_prob.
        Stops recursing beyond *depth* 5.
        """
        dist = await self._predict_one(prefix)
        children: dict[int, dict] = {}
        eligible = [
            i for i in range(len(dist))
            if dist[i] != 0 and prob_so_far * dist[i] >= min_prob
        ]
        # also break up the recursion if at any point it branches too eagerly
        if depth >= 5 or len(eligible) >= 3:
            return {"dist": dist, "children": children}
        if eligible:
            subtries = await asyncio.gather(*(
                self._predict_trie(
                    prefix + (tok,),
                    prob_so_far * dist[tok],
                    min_prob,
                    depth + 1,
                )
                for tok in eligible
            ))
            children = dict(zip(eligible, subtries))
        return {"dist": dist, "children": children}

    async def predict_batch(
        self, inputs: list[tuple[tuple[int, ...], float]]
    ) -> list[dict]:
        """Predict next-byte distribution tries for a batch of inputs.

        Each input is (prefix, min_prob).
        """
        if not inputs:
            return []

        return list(await asyncio.gather(*(
            self._predict_trie(ctx, 1.0, min_prob)
            for ctx, min_prob in inputs
        )))
