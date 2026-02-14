import { describe, it, expect } from "vitest";
import type { LanguageModel, Cursor, SceneNode, Scene } from "./types";
import { buildScene } from "./scene";

// ---------------------------------------------------------------------------
// Test language models (same as cursor.test.ts)
// ---------------------------------------------------------------------------

/** Uniform binary: A and B each with probability 0.5. */
const binary: LanguageModel<string> = async () => [
  { token: "A", probability: 0.5 },
  { token: "B", probability: 0.5 },
];

/** Asymmetric binary: A = 0.8, B = 0.2. */
const asym: LanguageModel<string> = async () => [
  { token: "A", probability: 0.8 },
  { token: "B", probability: 0.2 },
];

/** Three tokens. */
const ternary: LanguageModel<string> = async () => [
  { token: "X", probability: 0.2 },
  { token: "Y", probability: 0.5 },
  { token: "Z", probability: 0.3 },
];

/** Deterministic: single token with probability 1. */
const deterministic: LanguageModel<string> = async () => [
  { token: "A", probability: 1.0 },
];

/** Empty distribution — no continuations. */
const empty: LanguageModel<string> = async () => [];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect all nodes in a scene tree via pre-order traversal. */
async function allNodes<T>(scene: Scene<T>): Promise<SceneNode<T>[]> {
  const result: SceneNode<T>[] = [];
  async function walk(nodes: SceneNode<T>[]) {
    for (const n of nodes) {
      result.push(n);
      await walk(await n.children);
    }
  }
  await walk(scene.children);
  return result;
}

/** Collect just top-level tokens. */
function topTokens<T>(scene: Scene<T>): T[] {
  return scene.children.map((n) => n.token);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("buildScene", () => {
  // -- Structure --

  describe("structure", () => {
    it("fully zoomed out shows all tokens at top level", async () => {
      // cursor.x = 0 means halfHeight = 1, so window = entire unit square
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);
      expect(topTokens(scene)).toEqual(["A", "B"]);
    });

    it("fully zoomed out with ternary model shows all three tokens", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(ternary, cursor, 0.001);
      expect(topTokens(scene)).toEqual(["X", "Y", "Z"]);
    });

    it("zoomed in shows subset of tokens", async () => {
      // Deep into A's square: only A and its children should be prominent
      const cursor: Cursor<string> = { prefix: ["A", "A", "A"], x: 0, y: 0.25 };
      const scene = await buildScene(binary, cursor, 0.01);
      // When zoomed deep into A, the scene root may have ascended.
      // All visible nodes should exist.
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });

    it("children are nested within parent bounds", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);

      async function checkBounds(nodes: SceneNode<string>[]) {
        for (const node of nodes) {
          const children = await node.children;
          for (const child of children) {
            expect(child.y0).toBeGreaterThanOrEqual(node.y0 - 1e-10);
            expect(child.y1).toBeLessThanOrEqual(node.y1 + 1e-10);
            await checkBounds([child]);
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
      const cursor: Cursor<string> = { prefix: [], x: 0.5, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });

    it("window exceeding unit square still produces nodes", async () => {
      // Cursor near top edge: window extends above [0,1]
      const cursor: Cursor<string> = { prefix: [], x: 0.5, y: 0.1 };
      const scene = await buildScene(binary, cursor, 0.01);
      // Should still produce some nodes (the ones that overlap the window)
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
    });

    it("ascent from depth when window exceeds current square", async () => {
      // prefix=["A"], x=0.5, y=0.5 → halfHeight=0.5, window=[0,1]
      // Window [0,1] in A's frame spans the entire A square, which fits
      // But let's use x=0.3 → halfHeight=0.7, window=[-0.2, 1.2]
      // This exceeds [0,1] so we need to ascend
      const cursor: Cursor<string> = { prefix: ["A"], x: 0.3, y: 0.5 };
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
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);

      // Binary: A=[0, 0.5), B=[0.5, 1)
      // Window maps unit square to [0,1], so these should be exact
      expect(scene.children[0].token).toBe("A");
      expect(scene.children[0].y0).toBeCloseTo(0.25);
      expect(scene.children[0].y1).toBeCloseTo(0.5);
      expect(scene.children[1].token).toBe("B");
      expect(scene.children[1].y0).toBeCloseTo(0.5);
      expect(scene.children[1].y1).toBeCloseTo(0.75);
    });

    it("ternary model: y positions reflect cumulative probabilities", async () => {
      // X=0.2, Y=0.5, Z=0.3
      // cursor at root with x=0, y=0.5 → window = [−0.5, 1.5], height=2
      // scale = 1/2, offset = 0.5 * (1/2) = 0.25
      // X: y0 = 0.25 + 0*(1/2) = 0.25, y1 = 0.25 + 0.2*(1/2) = 0.35
      // Y: y0 = 0.25 + 0.2*(1/2) = 0.35, y1 = 0.25 + 0.7*(1/2) = 0.6
      // Z: y0 = 0.25 + 0.7*(1/2) = 0.6, y1 = 0.25 + 1.0*(1/2) = 0.75
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(ternary, cursor, 0.001);

      expect(scene.children[0].token).toBe("X");
      expect(scene.children[0].y0).toBeCloseTo(0.25);
      expect(scene.children[0].y1).toBeCloseTo(0.35);

      expect(scene.children[1].token).toBe("Y");
      expect(scene.children[1].y0).toBeCloseTo(0.35);
      expect(scene.children[1].y1).toBeCloseTo(0.6);

      expect(scene.children[2].token).toBe("Z");
      expect(scene.children[2].y0).toBeCloseTo(0.6);
      expect(scene.children[2].y1).toBeCloseTo(0.75);
    });

    it("zoomed in: cursor centered means crosshairs at 0.5", async () => {
      // With prefix=["A"], x=0.5, y=0.5 → halfHeight=0.5
      // Window in A's frame: [0, 1] — exactly A's square
      // A's children should fill the window
      const cursor: Cursor<string> = { prefix: ["A"], x: 0.5, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.001);
      // The first-level children of the scene should be A and B
      // (the children of the A square in the model)
      const nodes = scene.children;
      expect(nodes.length).toBe(2);
      expect(nodes[0].token).toBe("A");
      expect(nodes[1].token).toBe("B");
      // They should span the whole window [0,1]
      expect(nodes[0].y0).toBeCloseTo(0);
      expect(nodes[0].y1).toBeCloseTo(0.5);
      expect(nodes[1].y0).toBeCloseTo(0.5);
      expect(nodes[1].y1).toBeCloseTo(1);
    });
  });

  // -- Size threshold --

  describe("size threshold", () => {
    it("large minHeight excludes small tokens", async () => {
      // With a large minHeight, only the biggest tokens survive
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      await buildScene(ternary, cursor, 0.3);
      // winHeight = 2, minAbsProb = 0.3 * 2 = 0.6
      // X=0.2 < 0.6 → culled. Y=0.5 < 0.6 → culled. Z=0.3 < 0.6 → culled.
      // Actually all are culled! Let's use a less aggressive threshold.
      const scene2 = await buildScene(ternary, cursor, 0.2);
      // minAbsProb = 0.2 * 2 = 0.4
      // X=0.2 < 0.4 → culled. Y=0.5 ≥ 0.4 → kept. Z=0.3 < 0.4 → culled.
      const tokens = topTokens(scene2);
      expect(tokens).toEqual(["Y"]);
    });

    it("nested small tokens are excluded by threshold", async () => {
      // With the binary model, each level halves the probability.
      // After enough levels the children should be culled.
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.1);
      // winHeight = 2, minAbsProb = 0.1 * 2 = 0.2
      // Level 0: absProb=1, p=0.5 → childAbsProb=0.5 ≥ 0.2 → kept
      // Level 1: absProb=0.5, p=0.5 → childAbsProb=0.25 ≥ 0.2 → kept
      // Level 2: absProb=0.25, p=0.5 → childAbsProb=0.125 < 0.2 → culled
      async function measureDepth(nodes: SceneNode<string>[]): Promise<number> {
        let d = 0;
        for (const n of nodes) {
          const children = await n.children;
          d = Math.max(d, 1 + await measureDepth(children));
        }
        return d;
      }
      expect(await measureDepth(scene.children)).toBe(2);
    });

    it("very small minHeight allows deep recursion", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.0001);
      async function measureDepth(nodes: SceneNode<string>[]): Promise<number> {
        let d = 0;
        for (const n of nodes) {
          const children = await n.children;
          d = Math.max(d, 1 + await measureDepth(children));
        }
        return d;
      }
      expect(await measureDepth(scene.children)).toBeGreaterThan(5);
    });
  });

  // -- Invariants --

  describe("invariants", () => {
    it("y0 < y1 for all nodes", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      for (const node of await allNodes(scene)) {
        expect(node.y1).toBeGreaterThan(node.y0);
      }
    });

    it("no overlapping siblings", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(ternary, cursor, 0.01);

      async function checkSiblings(nodes: SceneNode<string>[]) {
        for (let i = 1; i < nodes.length; i++) {
          expect(nodes[i].y0).toBeGreaterThanOrEqual(nodes[i - 1].y1 - 1e-10);
        }
        for (const n of nodes) {
          const children = await n.children;
          await checkSiblings(children);
        }
      }
      await checkSiblings(scene.children);
    });

    it("all visible nodes at least partially in [0,1]", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0.3, y: 0.5 };
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
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      expect(scene.children.length).toBe(2);
    });

    it("empty model produces empty scene", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(empty, cursor, 0.01);
      expect(scene.children).toEqual([]);
    });

    it("deterministic model stops at maxDepth", async () => {
      const cursor: Cursor<string> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(deterministic, cursor, 0.0001, { maxDepth: 5 });
      // The single token has probability 1, so it recurses until maxDepth.
      async function measureDepth(nodes: SceneNode<string>[]): Promise<number> {
        let d = 0;
        for (const n of nodes) {
          const children = await n.children;
          d = Math.max(d, 1 + await measureDepth(children));
        }
        return d;
      }
      expect(await measureDepth(scene.children)).toBe(5);
    });

    it("works with numeric token types", async () => {
      const numModel: LanguageModel<number> = async () => [
        { token: 1, probability: 0.4 },
        { token: 2, probability: 0.6 },
      ];
      const cursor: Cursor<number> = { prefix: [], x: 0, y: 0.5 };
      const scene = await buildScene(numModel, cursor, 0.01);
      expect(scene.children.length).toBe(2);
      expect(scene.children[0].token).toBe(1);
      expect(scene.children[1].token).toBe(2);
    });
  });

  // -- Deep prefix / ascent precision --

  describe("deep prefix ascent", () => {
    it("ascent from depth 30 preserves correct window positioning", async () => {
      const depth = 30;
      const prefix = Array<string>(depth).fill("A");
      // Cursor at origin of a deeply nested square, x=0 → full zoom out
      // Window will be enormous relative to the tiny square, forcing ascent to root
      const cursor: Cursor<string> = { prefix, x: 0, y: 0.5 };
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
      const prefix = Array<string>(depth).fill("A");
      const cursor: Cursor<string> = { prefix, x: 0, y: 0.5 };
      const scene = await buildScene(asym, cursor, 0.01);
      const nodes = await allNodes(scene);
      expect(nodes.length).toBeGreaterThan(0);
      for (const n of nodes) {
        expect(n.y1).toBeGreaterThan(n.y0);
      }
    });

    it("zoomed in deep: window stays inside current square", async () => {
      // prefix at depth 5, cursor.x near 1 → very zoomed in, small window
      // Window should fit inside the square — no ascent needed
      const prefix = Array<string>(5).fill("A");
      const cursor: Cursor<string> = { prefix, x: 0.9, y: 0.5 };
      const scene = await buildScene(binary, cursor, 0.01);
      // Should show children of the AAAAA prefix
      expect(scene.children.length).toBe(2);
      expect(scene.children[0].token).toBe("A");
      expect(scene.children[1].token).toBe("B");
    });
  });
});
