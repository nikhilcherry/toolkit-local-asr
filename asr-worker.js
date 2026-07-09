/**
 * asr-worker.js — runs the actual Whisper pipeline off the main thread.
 * Owns the transformers.js pipeline instance and the job queue; the
 * main-thread wrapper (local-asr.js) only does postMessage plumbing.
 *
 * Model is swappable via the 'init' message's `model` field:
 *   - 'onnx-community/whisper-base'        (default, balanced)
 *   - 'distil-whisper/distil-small.en'     (lower latency, English-only)
 *   - 'onnx-community/whisper-small'       (use with language: null for
 *                                            Hinglish / code-switched audio)
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

// Model files are only ever fetched from the hub/CDN cache, never from a
// same-origin /models folder that doesn't exist in this standalone tool.
env.allowLocalModels = false;

/** @type {any} */
let transcriber = null;
/** @type {string|null} */
let language = 'english';
/** @type {number} */
let maxQueue = 2;

/** @type {{id: number, t0: number, t1: number, audio: Float32Array}[]} */
const queue = [];
let busy = false;

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'init') {
    handleInit(msg);
  } else if (msg.type === 'transcribe') {
    handleTranscribe(msg);
  }
};

/**
 * @param {{model: string, language: string|null, device: 'webgpu'|'wasm', maxQueue: number}} msg
 */
async function handleInit(msg) {
  language = msg.language;
  maxQueue = msg.maxQueue ?? 2;

  const model = msg.model;
  const requestedDevice = msg.device ?? 'webgpu';

  const progressCallback = (p) => {
    if (p.status === 'progress' && p.file) {
      const pct = p.total ? Math.round((p.loaded / p.total) * 100) : null;
      self.postMessage({
        type: 'status',
        text: pct != null ? `downloading ${p.file} (${pct}%)…` : `downloading ${p.file}…`,
      });
    } else if (p.status === 'done' && p.file) {
      self.postMessage({ type: 'status', text: `cached ${p.file}` });
    }
  };

  try {
    self.postMessage({ type: 'status', text: `loading model on ${requestedDevice}…` });
    transcriber = await pipeline('automatic-speech-recognition', model, {
      device: requestedDevice,
      progress_callback: progressCallback,
    });
  } catch (err) {
    if (requestedDevice !== 'webgpu') {
      self.postMessage({ type: 'init-error', error: err.message });
      return;
    }
    self.postMessage({
      type: 'status',
      text: `webgpu failed (${err.message}), falling back to wasm…`,
    });
    try {
      transcriber = await pipeline('automatic-speech-recognition', model, {
        device: 'wasm',
        progress_callback: progressCallback,
      });
    } catch (err2) {
      self.postMessage({ type: 'init-error', error: `wasm also failed: ${err2.message}` });
      return;
    }
  }

  self.postMessage({ type: 'status', text: 'warm' });
  self.postMessage({ type: 'ready' });
  processQueue();
}

/**
 * @param {{id: number, t0: number, t1: number, audio: Float32Array}} msg
 */
function handleTranscribe(msg) {
  // Drop the oldest QUEUED job (never the one currently running) once at
  // capacity, so a burst of chunks can't grow an ever-widening backlog.
  if (queue.length >= maxQueue) {
    const stale = queue.shift();
    self.postMessage({ type: 'dropped', t0: stale.t0, t1: stale.t1 });
  }
  queue.push({ id: msg.id, t0: msg.t0, t1: msg.t1, audio: msg.audio });
  processQueue();
}

async function processQueue() {
  if (busy || queue.length === 0 || !transcriber) return;
  busy = true;

  const job = queue.shift();
  const started = performance.now();
  try {
    const options = { task: 'transcribe' };
    if (language) options.language = language;

    const output = await transcriber(job.audio, options);
    const ms = Math.round(performance.now() - started);
    const text = (output?.text ?? '').trim();

    if (text.length > 0) {
      self.postMessage({ type: 'result', id: job.id, text, t0: job.t0, t1: job.t1, ms });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: job.id, error: err.message });
  } finally {
    busy = false;
    processQueue();
  }
}
