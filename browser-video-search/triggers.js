// Triggers: "tell me when you see a red shopping basket". Each frame of a live stream (the camera,
// or a video as it plays) is compared with each trigger's text, and a trigger fires when the frame
// stands out from what the stream has just been showing. The same rules as the glasses' own
// (backend app/glasses/scores.py), measured there on recorded footage:
//
// - A frame's plain similarity to a trigger says little: the floor moves with the scene (a whole
//   supermarket aisle raises every food-related trigger), so no fixed threshold holds. A frame's
//   score is its similarity minus the median similarity over the BASELINE_S before it: what has
//   just come into view. Something that stays in view stops scoring as the median catches up.
// - A trigger's level comes from its own noise: the median of its scores over the NOISE_S before,
//   plus SIGMAS robust standard deviations. It needs no labels and fits the scene. A trigger
//   waits for MIN_FRAMES frames before it can fire.
// - After one fires, nothing fires for QUIET_S. "Not this" raises its level above that score.
//
// No DOM and no model here, so `node --test` can check it (triggers.test.js).

import { cosine } from "./search.js";

export const BASELINE_S = 20;
export const NOISE_S = 60;
export const MIN_FRAMES = 10;
export const MIN_SPREAD = 0.005; // a covered lens has no spread; a level of 0 would always fire
export const SIGMAS = 3;
export const QUIET_S = 5; // after a firing: time to read the alert
export const MAX_LEVEL = 0.15; // a clear, close view scores 0.10 to 0.20
export const RAISE_BY = MIN_SPREAD; // "not this" puts the level this far above the score

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** One stream's recent frames and its triggers. Times are seconds on any steady clock. */
export class Watcher {
  constructor() {
    this.frames = []; // [{ t, vector }], oldest first
    this.triggers = []; // [{ id, text, vector, series: [{ t, value, score }], level, lastScore }]
    this.quietUntil = -Infinity;
    this.nextId = 1;
  }

  /** Start watching for `text`, whose embedding is `vector`. It is scored on the frames kept at
   * once, so it has a history and can fire sooner. */
  add(text, vector) {
    const trigger = { id: this.nextId++, text, vector, series: [], level: null, lastScore: null };
    for (const frame of this.frames) append(trigger.series, frame.t, cosine(frame.vector, vector));
    this.triggers.push(trigger);
    return trigger;
  }

  remove(id) {
    this.triggers = this.triggers.filter((trigger) => trigger.id !== id);
  }

  /** Forget the stream (a new video, a seek back): the triggers stay, their history goes. */
  reset() {
    this.frames = [];
    this.quietUntil = -Infinity;
    for (const trigger of this.triggers) {
      trigger.series = [];
      trigger.level = null;
      trigger.lastScore = null;
    }
  }

  /** Take in the frame seen at `t`; the trigger that fires on it, or null. */
  see(t, vector) {
    const oldest = t - NOISE_S - BASELINE_S;
    while (this.frames.length && this.frames[0].t < oldest) this.frames.shift();
    let fired = null;
    // The closest trigger first, as on the glasses.
    const ranked = this.triggers
      .map((trigger) => ({ trigger, value: cosine(vector, trigger.vector) }))
      .sort((a, b) => b.value - a.value);
    for (const { trigger, value } of ranked) {
      while (trigger.series.length && trigger.series[0].t < oldest) trigger.series.shift();
      trigger.level ??= levelOf(trigger.series, t);
      const score = append(trigger.series, t, value);
      trigger.lastScore = score;
      if (!fired && t >= this.quietUntil && trigger.level !== null && score >= trigger.level) {
        fired = { trigger, score };
      }
    }
    this.frames.push({ t, vector });
    if (fired) this.quietUntil = t + QUIET_S;
    return fired;
  }

  /** The visitor says a firing was wrong: the level goes above that score, within MAX_LEVEL. */
  notThis(id, score) {
    const trigger = this.triggers.find((each) => each.id === id);
    if (!trigger || trigger.level === null) return;
    const raised = Math.max(trigger.level, score + RAISE_BY);
    trigger.level = Math.min(raised, Math.max(trigger.level, MAX_LEVEL));
  }
}

/** The level from a trigger's scores before `t`, or null with fewer than MIN_FRAMES of them. */
export function levelOf(series, t, sigmas = SIGMAS) {
  const past = series.filter((s) => t - NOISE_S <= s.t && s.t < t).map((s) => s.score);
  if (past.length < MIN_FRAMES) return null;
  const middle = median(past);
  // 1.4826 times the median distance from the median is one standard deviation for normal
  // noise, and a real sighting among the scores hardly moves it.
  const spread = 1.4826 * median(past.map((score) => Math.abs(score - middle)));
  return middle + sigmas * Math.max(spread, MIN_SPREAD);
}

function append(series, t, value) {
  const before = series.filter((s) => t - BASELINE_S <= s.t && s.t < t).map((s) => s.value);
  const score = before.length ? value - median(before) : 0;
  series.push({ t, value, score });
  return score;
}
