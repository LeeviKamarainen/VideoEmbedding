# Visual search in the browser

Search a video, your photos or your live camera by describing what you saw ("the meat counter",
"a woman in a checked dress"), and ask to be told when something comes into view, with
[EmbeddingGemma 2](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) running on the
visitor's own GPU through WebGPU. There is no server: plain static files, which GitHub Pages can
host as they are. The model is downloaded from Hugging Face on first use and the browser keeps
it. Nothing the visitor picks or films leaves their device.

## What it does

- **Demo video.** A supermarket walk with its frames' embeddings ready in `data/index.json`, so
  searching it needs only the text part of the model (about 190 MB), which the page starts
  downloading as soon as it opens.
- **Your video.** Pick a video file and index it: frames are sampled every 0.5–2 seconds,
  embedded with the image encoder (about 100 MB more), and a frame much like the one before it
  is skipped. **Download index.json** saves the result so it can be hosted as the demo.
- **Your photos.** Pick any number of photos; each is embedded and can be searched.
- **Live camera.** Two frames a second are embedded and remembered while the camera runs (front or
  back on a phone), so you can search what it saw.
- **Watching for something.** "Tell me when I see… a red shopping basket" works on a playing
  video and on the live camera. An alert with the frame appears (and is spoken) when it comes
  into view; **Not this** raises the bar for that trigger.

## How it works

1. **Embedding.** A Web Worker (`worker.js`) runs EmbeddingGemma 2 with
   [Transformers.js](https://github.com/huggingface/transformers.js) 4.3.2 from jsDelivr: 4-bit
   weights (q4f16) on WebGPU, 8-bit on the CPU without it. Frames and questions become
   768-number vectors in one shared space. A question gets the model's query prompt
   (`task: search result | query: `); images take none.
2. **Search** (`search.js`). Cosine similarity, best first, one hit per 3 seconds of video so the
   results are different moments rather than one glance repeated.
3. **Triggers** (`triggers.js`), the same rules as the smart-glasses app this came from. Plain
   similarity says little, because the floor moves with the scene (a whole aisle of food raises
   every food trigger). So a frame's score is its similarity minus the median over the last 20
   seconds: what has just come into view. A trigger fires when that score passes a level set
   from its own noise over the last minute (median plus 3 robust standard deviations), which
   needs no tuning and fits whatever scene the camera is in. It waits for 10 frames before it
   can fire, and nothing fires for 5 seconds after an alert. On a video, the indexed frames
   are fed to the triggers as the playhead passes them, so watching costs no model work.
4. **Index file.** `{ model, video, credit?, examples?, watch_examples?, frames: [{ t, embedding }] }`,
   each embedding base64 of its float32 numbers, little-endian. `video` is relative to the index.

| What | Download (once) | Speed on an RTX 40-series GPU |
|---|---|---|
| Searching a ready-made index | ~190 MB | ~40 ms a search once loaded |
| Indexing a video, photos or the camera (adds the image encoder) | ~100 MB more | ~0.3–0.5 s a frame |

WebGPU is in Chrome and Edge on desktop and recent Android, Safari 26, and Firefox 141+ on
Windows. Without it everything still works on the CPU, slowly. The camera needs HTTPS, which
GitHub Pages has.

## Files

```
index.html          the page
style.css           light and dark themes
app.js              sources, indexing, searching, results, triggers and alerts
worker.js           EmbeddingGemma 2 in a Web Worker (text, and images when needed)
frames.js           frames out of a <video>: seek, draw, encode; thumbnails for the results
search.js           cosine ranking, near-duplicate skipping, the index file format
triggers.js         "tell me when I see…": scores against the recent scene, self-set levels
*.test.js           node --test search.test.js triggers.test.js
data/               the demo: supermarket.mp4 and its index.json
tools/              make-index.mjs: build an index.json in Node instead of the browser
```

## Run it locally

Double-clicking `index.html` does not work: from `file://` browsers block the page's JavaScript
modules and its `data/` files (the page says so). Serve the folder instead, with any static
server that supports HTTP Range requests (needed to seek in the video):

```bash
npx serve .
```

Python's `python -m http.server` does not send ranges, so the video cannot seek with it.

## Put it on GitHub Pages

1. Copy this folder into a repository (its root, or `docs/`).
2. Settings → Pages → Deploy from a branch, and pick that branch and folder.
3. Open `https://<user>.github.io/<repo>/`.

GitHub refuses files over 100 MB, so keep the demo video small. This one was made with
`ffmpeg -i original.mp4 -vf scale=-2:720,fps=30 -c:v libx264 -crf 31 -preset slow -movflags +faststart -an supermarket.mp4`
(141 MB at 1080p60 down to 22 MB).

## Change the demo

Either in the page: **Your video** → pick it → **Index this video** → **Download index.json**.

Or in Node, which does not need a browser tab left open (needs ffmpeg):

```bash
cd tools && npm install
node make-index.mjs ../data/my-video.mp4 ../data/index.json
```

Then edit `data/index.json`: set `credit` (`title`, `author`, `source`, `url`, `license`,
`license_url`), which the page shows under the video and in the footer, and `examples` and
`watch_examples`, the buttons under the search box. Update the footer's fallback text in
`index.html` too.

## Credits

- Demo video: [“Monipuolinen ostokokemus supermarketissa”](https://www.pexels.com/fi-fi/video/monipuolinen-ostokokemus-supermarketissa-39240106/)
  ("A varied shopping experience in a supermarket") by **Kiwi and Camera**, from Pexels, under the
  [Pexels License](https://www.pexels.com/license/). Re-encoded to 720p without changes to its
  content.
- Model: EmbeddingGemma 2 by Google; ONNX conversion by onnx-community; run with Transformers.js
  by Hugging Face.
