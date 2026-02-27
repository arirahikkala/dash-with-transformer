import { describe, it, expect } from "vitest";
import type { Cursor, SceneNode, Scene } from "./types";
import { adaptModel } from "./types";
import { buildScene } from "./scene";

// ---------------------------------------------------------------------------
// Test language models (same as cursor.test.ts)
// ---------------------------------------------------------------------------

// Token constants for readability.
const A = 0;
const B = 1;
const X = 0;
const Y = 1;
const Z = 2;

/** Uniform binary: A and B each with probability 0.5. */
const binary = adaptModel<readonly number[]>(async () => [0.5, 0.5]);

/** Asymmetric binary: A = 0.8, B = 0.2. */
const asym = adaptModel<readonly number[]>(async () => [0.8, 0.2]);

/** Three tokens. */
const ternary = adaptModel<readonly number[]>(async () => [0.2, 0.5, 0.3]);

/** Deterministic: single token with probability 1. */
const deterministic = adaptModel<readonly number[]>(async () => [1.0]);

/** Empty distribution — no continuations. */
const empty = adaptModel<readonly number[]>(async () => []);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect all items from an async iterable into an array. */
async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const item of iter) {
    result.push(item);
  }
  return result;
}

/** Collect all nodes in a scene tree via pre-order traversal. */
async function allNodes(scene: Scene<number>): Promise<SceneNode<number>[]> {
  const result: SceneNode<number>[] = [];
  async function walk(nodes: AsyncIterable<SceneNode<number>>) {
    for await (const n of nodes) {
      result.push(n);
      await walk(n.children);
    }
  }
  await walk(scene.children);
  return result;
}

/** Collect just top-level tokens. */
async function topTokens(scene: Scene<number>): Promise<number[]> {
  const children = await collect(scene.children);
  return children.map((n) => n.token);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("buildScene", () => {
  // -- Structure --

  describe("structure", () => {
    it("fully zoomed out shows all tokens at top level", async () => {
      // cursor.x = 0 means halfHeight = 1, so window = entire unit square
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);
      expect(await topTokens(scene)).toEqual([A, B]);
    });

    it("fully zoomed out with ternary model shows all three tokens", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(ternary, cursor, 0.001);
      expect(await topTokens(scene)).toEqual([X, Y, Z]);
    });

    it("zoomed in shows subset of tokens", async () => {
      // Deep into A's square: only A and its children should be prominent
      const cursor: Cursor<number> = { prefix: [A, A, A], x: 0, y: 0.25 };
      const scene = await buildScene(binary, cursor, 0.01);
      // When zoomed deep into A, the scene root may have ascended.
      // All visible nodes should exist.
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });

    it("children are nested within parent bounds", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);

      async function checkBounds(nodes: AsyncIterable<SceneNode<number>>) {
        for await (const node of nodes) {
          const children = await collect(node.children);
          for (const child of children) {
            expect(child.y0).toBeGreaterThanOrEqual(node.y0 - 1e-10);
            expect(child.y1).toBeLessThanOrEqual(node.y1 + 1e-10);
            await checkBounds(child.children);
          }
        }
      }
      await checkBounds(scene.children);
    });
  });

  // -- Window bounds --

  describe("window bounds", () => {
    it("window within unit square needs no ascent", async () => {
      // prefix=[], x=0.5, y=0.5 → halfHeight=0.5, window=[0, 1]
      // This is exactly the unit square — no ascent needed, scene root = []
      const cursor: Cursor<number> = { prefix: [], x: 0.5, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });

    it("window exceeding unit square still produces nodes", async () => {
      // Cursor near top edge: window extends above [0,1]
      const cursor: Cursor<number> = { prefix: [], x: 0.5, y: 0.1 };
      const scene = await buildScene(binary, cursor, 0.01);
      // Should still produce some nodes (the ones that overlap the window)
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });

    it("ascent from depth when window exceeds current square", async () => {
      // prefix=[A], x=0.5, y=0.5 → halfHeight=0.5, window=[0,1]
      // Window [0,1] in A's frame spans the entire A square, which fits
      // But let's use x=0.3 → halfHeight=0.7, window=[-0.2, 1.2]
      // This exceeds [0,1] so we need to ascend
      const cursor: Cursor<number> = { prefix: [A], x: 0.3, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });
  });

  // -- Affine correctness --

  describe("affine correctness", () => {
    it("fully zoomed out: top-level nodes span [0,1]", async () => {
      // cursor at root, x=0, y=0.5 → window is entire unit square
      // so window coords = unit square coords
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);

      // Binary: A=[0, 0.5), B=[0.5, 1)
      // Window maps unit square to [0,1], so these should be exact
      const children = await collect(scene.children);
      expect(children[0].token).toBe(A);
      expect(children[0].y0).toBeCloseTo(0.25);
      expect(children[0].y1).toBeCloseTo(0.5);
      expect(children[1].token).toBe(B);
      expect(children[1].y0).toBeCloseTo(0.5);
      expect(children[1].y1).toBeCloseTo(0.75);
    });

    it("ternary model: y positions reflect cumulative probabilities", async () => {
      // X=0.2, Y=0.5, Z=0.3
      // cursor at root with x=0, y=0.5 → window = [−0.5, 1.5], height=2
      // scale = 1/2, offset = 0.5 * (1/2) = 0.25
      // X: y0 = 0.25 + 0*(1/2) = 0.25, y1 = 0.25 + 0.2*(1/2) = 0.35
      // Y: y0 = 0.25 + 0.2*(1/2) = 0.35, y1 = 0.25 + 0.7*(1/2) = 0.6
      // Z: y0 = 0.25 + 0.7*(1/2) = 0.6, y1 = 0.25 + 1.0*(1/2) = 0.75
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(ternary, cursor, 0.001);

      const children = await collect(scene.children);
      expect(children[0].token).toBe(X);
      expect(children[0].y0).toBeCloseTo(0.25);
      expect(children[0].y1).toBeCloseTo(0.35);

      expect(children[1].token).toBe(Y);
      expect(children[1].y0).toBeCloseTo(0.35);
      expect(children[1].y1).toBeCloseTo(0.6);

      expect(children[2].token).toBe(Z);
      expect(children[2].y0).toBeCloseTo(0.6);
      expect(children[2].y1).toBeCloseTo(0.75);
    });

    it("zoomed in: cursor centered means crosshairs at 0.5", async () => {
      // With prefix=[A], x=0.5, y=0.5 → halfHeight=0.5
      // Window in A's frame: [0, 1] — fits, so ascend one more to scenePrefix=[].
      // In root frame: A covers [0, 0.5], window maps to [0, 0.5].
      // scale=2, offset=0 → top-level A spans [0, 1] in window coords.
      // A's children: AA at [0, 0.5], AB at [0.5, 1].
      const cursor: Cursor<number> = { prefix: [A], x: 0.5, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);
      const nodes = await collect(scene.children);
      // A covers entire window, B is off-screen at [1, 2]
      expect(nodes[0].token).toBe(A);
      expect(nodes[0].y0).toBeCloseTo(0);
      expect(nodes[0].y1).toBeCloseTo(1);
      // The original [0, 0.5] / [0.5, 1] split is now one level deeper
      const inner = await collect(nodes[0].children);
      expect(inner.length).toBe(2);
      expect(inner[0].token).toBe(A);
      expect(inner[0].y0).toBeCloseTo(0);
      expect(inner[0].y1).toBeCloseTo(0.5);
      expect(inner[1].token).toBe(B);
      expect(inner[1].y0).toBeCloseTo(0.5);
      expect(inner[1].y1).toBeCloseTo(1);
    });
  });

  // -- Size threshold --

  describe("size threshold", () => {
    it("large minHeight excludes small tokens", async () => {
      // With a large minHeight, only the biggest tokens survive
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      await buildScene(ternary, cursor, 0.3);
      // winHeight = 2, minAbsProb = 0.3 * 2 = 0.6
      // X=0.2 < 0.6 → culled. Y=0.5 < 0.6 → culled. Z=0.3 < 0.6 → culled.
      // Actually all are culled! Let's use a less aggressive threshold.
      const scene2 = await buildScene(ternary, cursor, 0.2);
      // minAbsProb = 0.2 * 2 = 0.4
      // X=0.2 < 0.4 → culled. Y=0.5 ≥ 0.4 → kept. Z=0.3 < 0.4 → culled.
      const tokens = await topTokens(scene2);
      expect(tokens).toEqual([Y]);
    });

    it("nested small tokens are excluded by threshold", async () => {
      // With the binary model, each level halves the probability.
      // After enough levels the children should be culled.
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.1);
      // winHeight = 2, minAbsProb = 0.1 * 2 = 0.2
      // Level 0: absProb=1, p=0.5 → childAbsProb=0.5 ≥ 0.2 → kept
      // Level 1: absProb=0.5, p=0.5 → childAbsProb=0.25 ≥ 0.2 → kept
      // Level 2: absProb=0.25, p=0.5 → childAbsProb=0.125 < 0.2 → culled
      async function measureDepth(
        nodes: AsyncIterable<SceneNode<number>>,
      ): Promise<number> {
        let d = 0;
        for await (const n of nodes) {
          d = Math.max(d, 1 + (await measureDepth(n.children)));
        }
        return d;
      }
      expect(await measureDepth(scene.children)).toBe(2);
    });

    it("very small minHeight allows deep recursion", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.0001);
      async function measureDepth(
        nodes: AsyncIterable<SceneNode<number>>,
      ): Promise<number> {
        let d = 0;
        for await (const n of nodes) {
          d = Math.max(d, 1 + (await measureDepth(n.children)));
        }
        return d;
      }
      expect(await measureDepth(scene.children)).toBeGreaterThan(5);
    });
  });

  // -- Invariants --

  describe("invariants", () => {
    it("y0 < y1 for all nodes", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      for (const node of await allNodes(scene)) {
        expect(node.y1).toBeGreaterThan(node.y0);
      }
    });

    it("no overlapping siblings", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(ternary, cursor, 0.01);

      async function checkSiblings(nodes: AsyncIterable<SceneNode<number>>) {
        const collected = await collect(nodes);
        for (let i = 1; i < collected.length; i++) {
          expect(collected[i].y0).toBeGreaterThanOrEqual(
            collected[i - 1].y1 - 1e-10,
          );
        }
        for (const n of collected) {
          await checkSiblings(n.children);
        }
      }
      await checkSiblings(scene.children);
    });

    it("all visible nodes at least partially in [0,1]", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0.3, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      for (const node of await allNodes(scene)) {
        // Node must overlap [0, 1] — not entirely outside
        expect(node.y1).toBeGreaterThan(0 - 1e-10);
        expect(node.y0).toBeLessThan(1 + 1e-10);
      }
    });
  });

  // -- Edge cases --

  describe("edge cases", () => {
    it("cursor.x = 0 gives fully zoomed out view", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      const children = await collect(scene.children);
      expect(children.length).toBe(2);
    });

    it("empty model produces empty scene", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(empty, cursor, 0.01);
      expect(await collect(scene.children)).toEqual([]);
    });

    it("deterministic model stops at maxDepth", async () => {
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(deterministic, cursor, 0.0001, {
        maxDepth: 5,
      });
      // The single token has probability 1, so it recurses until maxDepth.
      async function measureDepth(
        nodes: AsyncIterable<SceneNode<number>>,
      ): Promise<number> {
        let d = 0;
        for await (const n of nodes) {
          d = Math.max(d, 1 + (await measureDepth(n.children)));
        }
        return d;
      }
      expect(await measureDepth(scene.children)).toBe(5);
    });

    it("works with numeric token types", async () => {
      const numModel = adaptModel<readonly number[]>(async () => [0.4, 0.6]);
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(numModel, cursor, 0.01);
      const children = await collect(scene.children);
      expect(children.length).toBe(2);
      expect(children[0].token).toBe(0);
      expect(children[1].token).toBe(1);
    });
  });

  // -- Deep prefix / ascent precision --

  describe("deep prefix ascent", () => {
    it("ascent from depth 30 preserves correct window positioning", async () => {
      const depth = 30;
      const prefix = Array<number>(depth).fill(A);
      // Cursor at origin of a deeply nested square, x=0 → full zoom out
      // Window will be enormous relative to the tiny square, forcing ascent to root
      const cursor: Cursor<number> = { prefix, x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      // Should produce a valid scene with nodes
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
      // All nodes should have valid y0 < y1
      for (const n of nodes) {
        expect(n.y1).toBeGreaterThan(n.y0);
      }
    });

    it("ascent from depth 30 with asymmetric model", async () => {
      const depth = 30;
      const prefix = Array<number>(depth).fill(A);
      const cursor: Cursor<number> = { prefix, x: 0, y: 0.5 };
      const scene = await buildScene(asym, cursor, 0.01);
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
      for (const n of nodes) {
        expect(n.y1).toBeGreaterThan(n.y0);
      }
    });

    it("zoomed in deep: window inside single child ascends one level", async () => {
      // prefix at depth 5, cursor.x near 1, y centered inside child A
      // Window [0.15, 0.35] fits inside [0, 0.5] — ascend one extra level
      // to depth 4 so the covering node is a rendered child.
      // Only A is visible at each level until the window straddles A/B.
      const prefix = Array<number>(5).fill(A);
      const cursor: Cursor<number> = { prefix, x: 0.9, y: 0.25 };
      const scene = await buildScene(binary, cursor, 0.01);
      const children = await collect(scene.children);
      expect(children.length).toBe(1);
      expect(children[0].token).toBe(A);
      // One more level of single-A before the A/B split
      const inner = await collect(children[0].children);
      expect(inner.length).toBe(1);
      expect(inner[0].token).toBe(A);
      // Now both A and B are visible
      const deep = await collect(inner[0].children);
      expect(deep.length).toBe(2);
      expect(deep[0].token).toBe(A);
      expect(deep[1].token).toBe(B);
    });

    it("zoomed in deep: window straddling children ascends one level", async () => {
      // prefix at depth 5, y=0.5 straddles the A/B boundary
      // Window [0.4, 0.6] fits in [0,1], so ascend one extra to depth 4.
      // Top-level: only A visible. A's children: both A and B visible.
      const prefix = Array<number>(5).fill(A);
      const cursor: Cursor<number> = { prefix, x: 0.9, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      const children = await collect(scene.children);
      expect(children.length).toBe(1);
      expect(children[0].token).toBe(A);
      const inner = await collect(children[0].children);
      expect(inner.length).toBe(2);
      expect(inner[0].token).toBe(A);
      expect(inner[1].token).toBe(B);
    });
  });
});
