// Build data/index.json for a video outside the browser: the same model, the same frame rule and
// the same file format as the page's "Index this video", but in Node on the GPU (WebGPU) with
// ffmpeg pulling the frames, so a long video does not need a browser tab left open.
//
//   cd tools && npm install
//   node make-index.mjs ../data/supermarket.mp4 ../data/index.json [--every 0.5] [--device cpu]
//
// Needs ffmpeg on the PATH. Without a GPU, --device cpu works but is about ten times slower.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, relative, dirname } from "node:path";
import { parseArgs } from "node:util";

import { AutoConfig, AutoModel, AutoProcessor, RawImage } from "@huggingface/transformers";

import { toIndex, worthKeeping } from "../search.js";

const MODEL = "onnx-community/embeddinggemma-2-ONNX";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    every: { type: "string", default: "0.5" },
    device: { type: "string", default: "webgpu" },
    // fp16: in Node it is ~6x faster than the page's 4-bit q4f16 (0.25 s against 1.6 s a frame
    // on an RTX 40-series), and the two agree to 0.97+ cosine, so either searches the same.
    dtype: { type: "string", default: "fp16" },
  },
});
const [video, out = "index.json"] = positionals;
if (!video) {
  console.error("Usage: node make-index.mjs <video> [index.json] [--every 0.5] [--device cpu]");
  process.exit(1);
}
const every = Number(values.every);

// The frames, as the page samples them: one every `every` seconds, at most 768 pixels a side.
const folder = mkdtempSync(join(tmpdir(), "frames-"));
console.log(`Extracting a frame every ${every} s from ${video}…`);
execFileSync("ffmpeg", [
  "-v", "error", "-i", video,
  "-vf", `fps=1/${every},scale='min(768,iw)':'min(768,ih)':force_original_aspect_ratio=decrease`,
  "-q:v", "3", join(folder, "%05d.jpg"),
]);
const files = readdirSync(folder).filter((name) => name.endsWith(".jpg")).sort();

console.log(`Loading ${MODEL} (${values.dtype}) on ${values.device}…`);
const config = await AutoConfig.from_pretrained(MODEL);
config.audio_config = null;
const [processor, model] = await Promise.all([
  AutoProcessor.from_pretrained(MODEL),
  AutoModel.from_pretrained(MODEL, {
    config,
    device: values.device,
    dtype: { model: values.dtype, vision_encoder: values.dtype },
  }),
]);

const kept = [];
const started = performance.now();
for (const [i, name] of files.entries()) {
  const image = await RawImage.fromBlob(new Blob([readFileSync(join(folder, name))]));
  const { sentence_embedding } = await model(await processor(null, image));
  const frame = { t: i * every, vector: Float32Array.from(sentence_embedding.data) };
  if (worthKeeping(frame, kept.at(-1))) kept.push(frame);
  if ((i + 1) % 20 === 0 || i + 1 === files.length) {
    const perFrame = (performance.now() - started) / (i + 1);
    console.log(`${i + 1} / ${files.length} frames, ${kept.length} kept, ${Math.round(perFrame)} ms each`);
  }
}
rmSync(folder, { recursive: true, force: true });

// The page finds the video relative to the index.
const videoPath = relative(dirname(out), video).replaceAll("\\", "/") || basename(video);
writeFileSync(out, JSON.stringify(toIndex({ model: MODEL, video: videoPath, frames: kept })));
console.log(`Wrote ${out}: ${kept.length} frames of ${videoPath}`);
