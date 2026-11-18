// Pulling still frames out of a <video> in the browser: seek, draw onto a canvas, encode.

/** Seek and wait until the frame at `t` is ready to draw. */
export function seek(video, t) {
  return new Promise((resolve, reject) => {
    if (Math.abs(video.currentTime - t) < 1e-3 && video.readyState >= 2) return resolve();
    const done = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", failed);
      resolve();
    };
    const failed = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", failed);
      reject(new Error("The video could not be read"));
    };
    video.addEventListener("seeked", done);
    video.addEventListener("error", failed);
    video.currentTime = t;
  });
}

/** Wait until a video knows its duration and size. */
export function ready(video) {
  if (video.readyState >= 1 && video.duration) return Promise.resolve();
  return new Promise((resolve, reject) => {
    video.addEventListener("loadedmetadata", () => resolve(), { once: true });
    video.addEventListener("error", () => reject(new Error("The video could not be opened")), {
      once: true,
    });
  });
}

/** A video's frame on screen, or a canvas, as a canvas at most `maxSide` pixels wide or tall. */
export function draw(source, maxSide) {
  const width = source.videoWidth ?? source.width;
  const height = source.videoHeight ?? source.height;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export function toBlob(canvas, type = "image/jpeg", quality = 0.9) {
  return new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

/** Every `everyS` seconds of the video as { t, canvas }, in order. The model shrinks images
 * itself, so 768 pixels is plenty. Stops early when `signal` aborts. */
export async function* sample(video, everyS, { maxSide = 768, signal } = {}) {
  await ready(video);
  for (let t = 0; t < video.duration; t += everyS) {
    if (signal?.aborted) return;
    await seek(video, t);
    yield { t, canvas: draw(video, maxSide) };
  }
}

/** Small JPEG data URLs of a video's frames, one at a time, for result cards. */
export function thumbnails(src, maxSide = 240) {
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  video.src = src;
  const cache = new Map();
  let queue = Promise.resolve();
  return {
    /** Keep a frame already drawn (while indexing), so its card needs no seek. */
    put(t, canvas) {
      cache.set(t, draw(canvas, maxSide).toDataURL("image/jpeg", 0.8));
    },
    get(t) {
      if (cache.has(t)) return Promise.resolve(cache.get(t));
      queue = queue.then(async () => {
        if (cache.has(t)) return cache.get(t);
        await ready(video);
        await seek(video, t);
        const url = draw(video, maxSide).toDataURL("image/jpeg", 0.8);
        cache.set(t, url);
        return url;
      });
      return queue;
    },
  };
}
