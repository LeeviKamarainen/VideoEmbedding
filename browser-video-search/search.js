// Searching a video's frames by their embeddings, and the index file that holds them. No DOM and
// no model here, so `node --test` can check it (search.test.js).

/** Frames closer in time than this show the same moment: the best few hits would otherwise often
 * be one glance, sampled twice a second. */
export const SAME_MOMENT_S = 3;
/** While indexing, a frame this close (cosine distance) to the last one kept, and less than
 * KEEP_EVERY_S after it, adds nothing and is skipped: the glasses' own rule for a stream. */
export const SCENE_CHANGE = 0.05;
export const KEEP_EVERY_S = 10;

/** Cosine similarity. The model's vectors are unit length, but nothing here relies on it. */
export function cosine(a, b) {
  if (a.length !== b.length) {
    throw new Error(`Embeddings of ${a.length} and ${b.length} numbers cannot be compared`);
  }
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa && bb ? dot / Math.sqrt(aa * bb) : 0;
}

/** The `k` frames closest to the query, best first, at most one per `sameMoment` seconds (0 for
 * photos, which are each their own moment). `frames` is [{ t, vector, ... }], t in seconds; a
 * hit keeps its frame's other fields. */
export function search(query, frames, k = 6, sameMoment = SAME_MOMENT_S) {
  const scored = frames
    .map((frame) => ({ ...frame, vector: undefined, score: cosine(query, frame.vector) }))
    .sort((a, b) => b.score - a.score);
  const hits = [];
  for (const hit of scored) {
    if (hits.length >= k) break;
    if (hits.some((kept) => Math.abs(kept.t - hit.t) < sameMoment)) continue;
    hits.push(hit);
  }
  return hits;
}

/** Whether a new frame is worth keeping after `last` ({ t, vector } or null). */
export function worthKeeping(frame, last) {
  if (!last) return true;
  return frame.t - last.t >= KEEP_EVERY_S || 1 - cosine(frame.vector, last.vector) >= SCENE_CHANGE;
}

/** "1:05" */
export function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// ---- The index file: { model, video, frames: [{ t, embedding }] }, each embedding base64 of its
// float32 numbers, little-endian (a third of the size of a JSON list).

export function encodeVector(vector) {
  const bytes = new Uint8Array(Float32Array.from(vector).buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

export function decodeVector(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer); // little-endian, as every browser's platform is
}

export function toIndex({ model, video, frames }) {
  return {
    model,
    video,
    frames: frames.map((f) => ({ t: Math.round(f.t * 1000) / 1000, embedding: encodeVector(f.vector) })),
  };
}

export function fromIndex(index) {
  if (!Array.isArray(index?.frames)) throw new Error("Not an index: it has no frames");
  return index.frames
    .map((f) => ({ t: f.t, vector: decodeVector(f.embedding) }))
    .sort((a, b) => a.t - b.t);
}
