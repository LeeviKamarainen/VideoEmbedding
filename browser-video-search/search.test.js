// node --test search.test.js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  clock,
  cosine,
  decodeVector,
  encodeVector,
  fromIndex,
  SAME_MOMENT_S,
  search,
  toIndex,
  worthKeeping,
} from "./search.js";

const v = (...xs) => Float32Array.from(xs);

test("cosine ignores length and rejects other widths", () => {
  assert.equal(cosine(v(1, 0), v(3, 0)), 1);
  assert.equal(cosine(v(1, 0), v(0, 2)), 0);
  assert.throws(() => cosine(v(1, 0), v(1, 0, 0)), /2 and 3/);
});

test("search ranks best first, one frame per moment", () => {
  const frames = [
    { t: 5, vector: v(1, 0) },
    { t: 5.5, vector: v(1, 0.05) }, // the same glance as t=5
    { t: 5 + SAME_MOMENT_S, vector: v(0.5, 0.5) },
    { t: 30, vector: v(0, 1) },
  ];
  assert.deepEqual(
    search(v(1, 0), frames).map((hit) => hit.t),
    [5, 8, 30],
  );
  assert.deepEqual(
    search(v(1, 0), frames, 1).map((hit) => hit.t),
    [5],
  );
  // Photos: every one is its own moment.
  assert.deepEqual(
    search(v(1, 0), frames, 6, 0).map((hit) => hit.t),
    [5, 5.5, 8, 30],
  );
});

test("a hit keeps its frame's fields, without the vector", () => {
  const [hit] = search(v(1, 0), [{ t: 0, vector: v(1, 0), label: "a.jpg" }]);
  assert.equal(hit.label, "a.jpg");
  assert.equal(hit.vector, undefined);
});

test("worthKeeping skips a frame like the last one, unless the last one is old", () => {
  const last = { t: 0, vector: v(1, 0) };
  assert.equal(worthKeeping(last, null), true);
  assert.equal(worthKeeping({ t: 0.5, vector: v(1, 0.01) }, last), false);
  assert.equal(worthKeeping({ t: 0.5, vector: v(0, 1) }, last), true);
  assert.equal(worthKeeping({ t: 10, vector: v(1, 0) }, last), true);
});

test("vectors and the index survive a round trip", () => {
  const vector = v(0.25, -1.5, 3);
  assert.deepEqual(decodeVector(encodeVector(vector)), vector);
  const index = toIndex({ model: "m", video: "a.mp4", frames: [{ t: 2.0004, vector }] });
  assert.equal(index.frames[0].t, 2);
  assert.deepEqual(fromIndex(JSON.parse(JSON.stringify(index))), [{ t: 2, vector }]);
  assert.throws(() => fromIndex({}), /no frames/);
});

test("clock", () => {
  assert.equal(clock(65.9), "1:05");
  assert.equal(clock(0), "0:00");
});
