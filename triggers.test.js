// node --test triggers.test.js
import assert from "node:assert/strict";
import { test } from "node:test";

import { BASELINE_S, levelOf, MAX_LEVEL, MIN_FRAMES, MIN_SPREAD, QUIET_S, SIGMAS, Watcher } from "./triggers.js";

// A unit vector at `similarity` to [1, 0]: the trigger's text in these tests.
const at = (similarity) => Float32Array.from([similarity, Math.sqrt(1 - similarity ** 2)]);
const TEXT = Float32Array.from([1, 0]);

/** Two frames a second of slightly noisy background at 0.2 similarity, from `from` for `n`. */
function background(watcher, from, n) {
  let fired = null;
  for (let i = 0; i < n; i++) {
    fired ??= watcher.see(from + i / 2, at(0.2 + (i % 3) * 0.004));
  }
  return fired;
}

test("a trigger waits for MIN_FRAMES before it has a level", () => {
  const watcher = new Watcher();
  const trigger = watcher.add("basket", TEXT);
  background(watcher, 0, MIN_FRAMES);
  assert.equal(trigger.level, null);
  watcher.see(MIN_FRAMES / 2, at(0.9)); // a clear sighting, but no level yet to reach
  assert.notEqual(trigger.level, null);
});

test("fires when something comes into view, not while the scene stays the same", () => {
  const watcher = new Watcher();
  const trigger = watcher.add("basket", TEXT);
  assert.equal(background(watcher, 0, 40), null);
  const fired = watcher.see(20, at(0.3));
  assert.equal(fired?.trigger, trigger);
  assert.ok(fired.score > trigger.level);
  // The same view a moment later is within the quiet time.
  assert.equal(watcher.see(20.5, at(0.3)), null);
  // Once it has been in view for half the baseline, the median has caught up: no more firings.
  const firings = [];
  for (let t = 21; t < 60; t += 0.5) if (watcher.see(t, at(0.3))) firings.push(t);
  assert.ok(firings.every((t) => t < 20 + BASELINE_S / 2 + QUIET_S), `fired at ${firings}`);
});

test("a trigger added later is scored on the frames already seen", () => {
  const watcher = new Watcher();
  background(watcher, 0, 30);
  const trigger = watcher.add("basket", TEXT);
  assert.equal(trigger.series.length, 30);
  assert.equal(watcher.see(15, at(0.4))?.trigger, trigger);
});

test("notThis raises the level above the score, within MAX_LEVEL", () => {
  const watcher = new Watcher();
  const trigger = watcher.add("basket", TEXT);
  background(watcher, 0, 40);
  const fired = watcher.see(20, at(0.3));
  watcher.notThis(trigger.id, fired.score);
  assert.ok(trigger.level >= Math.min(fired.score + MIN_SPREAD, MAX_LEVEL) - 1e-9);
  assert.ok(trigger.level <= Math.max(MAX_LEVEL, fired.score + MIN_SPREAD));
});

test("levelOf never sits closer than SIGMAS * MIN_SPREAD above flat noise", () => {
  const flat = Array.from({ length: 20 }, (_, i) => ({ t: i, value: 0.2, score: 0 }));
  assert.equal(levelOf(flat, 20), SIGMAS * MIN_SPREAD);
  assert.equal(levelOf(flat.slice(0, MIN_FRAMES - 1), 20), null);
});

test("reset forgets the stream but keeps the triggers", () => {
  const watcher = new Watcher();
  const trigger = watcher.add("basket", TEXT);
  background(watcher, 0, 30);
  watcher.reset();
  assert.equal(watcher.triggers.length, 1);
  assert.equal(trigger.level, null);
  assert.equal(trigger.series.length, 0);
});
