// EmbeddingGemma 2 in the browser, off the page's thread. It turns video frames and search
// questions into vectors in one shared 768-number space, so a question can be compared with every
// frame. Transformers.js downloads the model from Hugging Face the first time and the browser
// keeps it (Cache API): nothing about the video or the questions leaves the device.
//
// Messages in:  { id, type: "load", vision }      load the model (vision: also the image encoder)
//               { id, type: "text", text }        embed a search question
//               { id, type: "image", blob }       embed one frame (a JPEG/PNG Blob)
// Messages out: { id, type: "progress", loaded, total }   while downloading
//               { id, type: "ready", device, dtype }
//               { id, type: "vector", vector }            a Float32Array, unit length
//               { id, type: "error", message }

import {
  AutoConfig,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";

const MODEL = "onnx-community/embeddinggemma-2-ONNX";

// EmbeddingGemma 2 puts a prompt before a search query (its sentence-transformers prompt
// "SearchQuery", from the model's config_sentence_transformers.json). Images take none. A query
// embedded without it lands elsewhere in the space and matches worse.
const QUERY_PROMPT = "task: search result | query: ";

let gpuPromise = null;
let loaded = null; // { vision, promise }: the model as last loaded

/** WebGPU when the browser has an adapter, and whether it does 16-bit floats (q4f16 needs it). */
function gpu() {
  gpuPromise ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
    return adapter
      ? { device: "webgpu", f16: adapter.features.has("shader-f16") }
      : { device: "wasm", f16: false };
  })();
  return gpuPromise;
}

/** 4-bit weights on the GPU (0.988 of fp32 for text, 0.975 for images, by the model card),
 * 8-bit on the CPU. */
function dtypeFor({ device, f16 }) {
  if (device === "wasm") return "q8";
  return f16 ? "q4f16" : "q4";
}

/** The model, with the image encoder when `vision`. Searching a ready-made index only needs the
 * text part (about 190 MB); embedding a video needs the image encoder too (about 100 MB more). */
function model(id, vision) {
  if (loaded && (loaded.vision || !vision)) return loaded.promise;
  const previous = loaded?.promise;
  const promise = (async () => {
    const where = await gpu();
    const dtype = dtypeFor(where);
    const files = new Map();
    const progress_callback = (info) => {
      if (info.status !== "progress") return;
      files.set(info.file, info);
      let done = 0;
      let total = 0;
      for (const file of files.values()) {
        done += file.loaded;
        total += file.total;
      }
      self.postMessage({ id, type: "progress", loaded: done, total });
    };
    const config = await AutoConfig.from_pretrained(MODEL);
    config.audio_config = null; // no audio here
    if (!vision) config.vision_config = null;
    const [processor, net] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL, { progress_callback }),
      AutoModel.from_pretrained(MODEL, {
        config,
        device: where.device,
        dtype: { model: dtype, vision_encoder: dtype },
        progress_callback,
      }),
    ]);
    // A text-only copy loaded before is replaced, not kept beside this one in GPU memory.
    previous?.then(({ net: old }) => old.dispose?.()).catch(() => {});
    return { processor, net, device: where.device, dtype };
  })();
  loaded = { vision, promise };
  promise.catch(() => {
    if (loaded?.promise === promise) loaded = null; // let a retry start over
  });
  return promise;
}

async function embed(inputs, net) {
  // Mean-pooled and L2-normalised by the model itself.
  const { sentence_embedding } = await net(inputs);
  return Float32Array.from(sentence_embedding.data);
}

async function embedText(id, text) {
  const { processor, net } = await model(id, false);
  return embed(processor.tokenizer(QUERY_PROMPT + text, { truncation: true }), net);
}

async function embedImage(id, blob) {
  const { processor, net } = await model(id, true);
  const image = await RawImage.fromBlob(blob);
  return embed(await processor(null, image), net); // one image, one sample
}

self.onmessage = async ({ data }) => {
  const { id } = data;
  try {
    if (data.type === "load") {
      const { device, dtype } = await model(id, data.vision);
      self.postMessage({ id, type: "ready", device, dtype });
    } else if (data.type === "text") {
      const vector = await embedText(id, data.text);
      self.postMessage({ id, type: "vector", vector }, [vector.buffer]);
    } else if (data.type === "image") {
      const vector = await embedImage(id, data.blob);
      self.postMessage({ id, type: "vector", vector }, [vector.buffer]);
    }
  } catch (problem) {
    self.postMessage({ id, type: "error", message: problem?.message ?? String(problem) });
  }
};
