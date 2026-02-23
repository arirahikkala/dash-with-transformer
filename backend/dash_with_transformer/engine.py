from __future__ import annotations

import asyncio
import os
from collections import OrderedDict

import numpy as np
from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams, ByteBeamState
from genlm.bytes.byte_lm.trie_state import TrieMode

BEAM_K = int(os.environ.get("BEAM_K", "5"))
PRUNE_THRESHOLD = float(os.environ.get("PRUNE_THRESHOLD", "0.05"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2-medium")
CACHE_MAX_SIZE = int(os.environ.get("CACHE_MAX_SIZE", "10000"))


class BytePredictionEngine:
    def __init__(self):
        self._cache: OrderedDict[bytes, ByteBeamState] = OrderedDict()
        self._lock = asyncio.Lock()

    async def start(self):
        llm = load_model_by_name(MODEL_NAME, backend="vllm")
        params = BeamParams(K=BEAM_K, prune_threshold=PRUNE_THRESHOLD)
        initial = await ByteBeamState.initial(llm, params)
        # Store the initial state in WITHOUT_EOS mode for prefix caching.
        # prefill() internally operates in WITHOUT_EOS, so all cached states
        # use this mode. We switch to WITH_EOS only at query time.
        self._cache[b""] = initial.with_mode(TrieMode.WITHOUT_EOS)
        self._llm = llm
        self._initial = initial

    async def shutdown(self):
        await self._initial.cleanup()

    def _longest_cached_prefix(self, ctx: bytes) -> tuple[bytes, ByteBeamState]:
        """Find the longest prefix of ctx that exists in the cache."""
        for i in range(len(ctx), -1, -1):
            prefix = ctx[:i]
            if prefix in self._cache:
                # Move to end for LRU
                self._cache.move_to_end(prefix)
                return prefix, self._cache[prefix]
        # Should never happen since b"" is always in cache
        raise RuntimeError("empty prefix missing from cache")

    def _put_cache(self, key: bytes, state: ByteBeamState):
        """Insert into the cache with LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        self._cache[key] = state
        while len(self._cache) > CACHE_MAX_SIZE:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            if evicted_key == b"":
                # Never evict the empty-prefix sentinel; re-insert at front and stop.
                self._cache[evicted_key] = evicted_val
                self._cache.move_to_end(evicted_key, last=False)
                break

    async def _predict_one(self, ctx: bytes) -> list[float]:
        """Get next-byte probabilities for a single byte context."""
        async with self._lock:
            prefix, state = self._longest_cached_prefix(ctx)

        # Advance byte-by-byte through the uncached suffix
        suffix = ctx[len(prefix):]
        for i, byte_val in enumerate(suffix):
            state = await (state.prune() << byte_val)
            new_prefix = ctx[: len(prefix) + i + 1]
            async with self._lock:
                self._put_cache(new_prefix, state)

        # Switch to WITH_EOS for generation and get log-probs
        gen_state = state.with_mode(TrieMode.WITH_EOS)
        logp = await gen_state.logp_next()
        probs = np.exp(logp.ps[:256]).tolist()
        return probs

    async def _predict_trie(
        self,
        prefix: bytes,
        prob_so_far: float,
        min_prob: float,
        depth: int = 0,
    ) -> dict:
        """Recursively build a trie of next-byte distributions.

        Only expands children whose probability >= min_prob.
        Stops recursing beyond *depth* 5.
        """
        dist = await self._predict_one(prefix)
        children: dict[int, dict] = {}
        eligible = [
            b for b in range(256)
            if dist[b] != 0 and prob_so_far * dist[b] >= min_prob
        ]
        # also break up the recursion if at any point it branches too eagerly
        if depth >= 5 or len(eligible) >= 3:
            return {"dist": dist, "children": children}
        if eligible:
            subtries = await asyncio.gather(*(
                self._predict_trie(
                    prefix + bytes([b]),
                    prob_so_far * dist[b],
                    min_prob,
                    depth + 1,
                )
                for b in eligible
            ))
            children = dict(zip(eligible, subtries))
        return {"dist": dist, "children": children}

    async def predict_batch(
        self, inputs: list[tuple[bytes, float]]
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
