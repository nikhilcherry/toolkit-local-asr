/**
 * LocalASR — in-browser Whisper transcription via transformers.js, running
 * entirely on-device in a Web Worker. Single-file main-thread wrapper,
 * zero dependencies. transformers.js itself loads from CDN inside the
 * worker.
 *
 * Part of a hackathon toolkit: MicVAD's 'chunk' event feeds transcribe()
 * directly; the t0/t1 on each 'result' feed straight into
 * AttributionFuser.attribute(t0, t1). Works completely standalone — no
 * imports of other toolkit tools.
 */

/**
 * Minimal inline EventEmitter — no dependencies.
 */
class EventEmitter {
  constructor() {
    /** @type {Map<string, Set<Function>>} */
    this._listeners = new Map();
  }

  /**
   * @param {string} event
   * @param {Function} handler
   * @returns {this}
   */
  on(event, handler) {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event).add(handler);
    return this;
  }

  /**
   * @param {string} event
   * @param {Function} handler
   * @returns {this}
   */
  off(event, handler) {
    const set = this._listeners.get(event);
    if (set) set.delete(handler);
    return this;
  }

  /**
   * @param {string} event
   * @param {*} [payload]
   */
  emit(event, payload) {
    const set = this._listeners.get(event);
    if (!set || set.size === 0) return;
    for (const handler of Array.from(set)) handler(payload);
  }
}

/**
 * @typedef {Object} LocalASROptions
 * @property {string} [model='onnx-community/whisper-base'] - any transformers.js ASR model id.
 * @property {string|null} [language='english'] - forced transcription language, or null to auto-detect.
 * @property {'webgpu'|'wasm'} [device='webgpu'] - preferred compute backend; falls back to 'wasm' automatically.
 * @property {number} [maxQueue=2] - max queued (not-yet-running) jobs before the oldest is dropped.
 * @property {string} [workerUrl='./asr-worker.js'] - path to the worker script.
 */

/**
 * @typedef {Object} ResultEvent
 * @property {string} text - transcribed text (never empty/whitespace-only).
 * @property {number} t0 - start bound passed to transcribe(), echoed back.
 * @property {number} t1 - end bound passed to transcribe(), echoed back.
 * @property {number} ms - inference time in milliseconds.
 */

/**
 * @typedef {Object} DroppedEvent
 * @property {number} t0 - t0 of the discarded job.
 * @property {number} t1 - t1 of the discarded job.
 */

export class LocalASR extends EventEmitter {
  /**
   * @param {LocalASROptions} [options]
   */
  constructor(options = {}) {
    super();

    /** @type {string} */
    this.model = options.model ?? 'onnx-community/whisper-base';
    /** @type {string|null} */
    this.language = options.language === undefined ? 'english' : options.language;
    /** @type {'webgpu'|'wasm'} */
    this.device = options.device ?? 'webgpu';
    /** @type {number} */
    this.maxQueue = options.maxQueue ?? 2;
    /** @type {string} */
    this.workerUrl = options.workerUrl ?? './asr-worker.js';

    this._worker = null;
    this._isReady = false;
    this._initPromise = null;
    this._rejectInit = null;
    this._idCounter = 0;
  }

  /** @returns {boolean} true once the model is downloaded/cached and warm. */
  get isReady() {
    return this._isReady;
  }

  /**
   * Spins up the worker, loads the model (from cache after the first run),
   * and warms it up. WebGPU failures fall back to wasm automatically and
   * only surface as a 'status' event; init() rejects only if both backends
   * fail.
   * @returns {Promise<void>}
   */
  init() {
    if (this._initPromise) return this._initPromise;

    this._initPromise = new Promise((resolve, reject) => {
      this._rejectInit = reject;
      const worker = new Worker(this.workerUrl, { type: 'module' });
      this._worker = worker;

      worker.onmessage = (event) => this._handleMessage(event.data, resolve, reject);
      worker.onerror = (err) => {
        const message = `LocalASR: worker error: ${err.message}`;
        // After init has settled, reject() is a no-op — surface crashes as
        // a status event so a running app still learns the worker died.
        this.emit('status', message);
        reject(new Error(message));
      };

      worker.postMessage({
        type: 'init',
        model: this.model,
        language: this.language,
        device: this.device,
        maxQueue: this.maxQueue,
      });
    });

    // A failed load (CDN/model download failure, WebGPU+wasm both broken)
    // must not be cached forever: tear down the dead worker and clear the
    // promise so a later init() can retry from scratch.
    this._initPromise.catch(() => {
      this._rejectInit = null;
      if (!this._isReady) {
        if (this._worker) {
          this._worker.terminate();
          this._worker = null;
        }
        this._initPromise = null;
      }
    });

    return this._initPromise;
  }

  /**
   * @param {any} msg
   * @param {Function} resolve
   * @param {Function} reject
   */
  _handleMessage(msg, resolve, reject) {
    switch (msg.type) {
      case 'status':
        this.emit('status', msg.text);
        break;
      case 'ready':
        this._isReady = true;
        this._rejectInit = null;
        resolve();
        break;
      case 'init-error':
        reject(new Error(`LocalASR: ${msg.error}`));
        break;
      case 'result':
        this.emit('result', { text: msg.text, t0: msg.t0, t1: msg.t1, ms: msg.ms });
        break;
      case 'dropped':
        this.emit('dropped', { t0: msg.t0, t1: msg.t1 });
        break;
      case 'error':
        this.emit('status', `transcription error: ${msg.error}`);
        break;
    }
  }

  /**
   * Queues audio for transcription. Fire-and-forget — the result (or a
   * 'dropped' event if this job gets evicted for staleness) arrives later
   * via events. If the worker already has maxQueue jobs waiting, the
   * oldest queued job (never the one currently running) is dropped to make
   * room, so live captions never fall further behind than maxQueue chunks.
   * @param {Float32Array} float32Audio - mono PCM samples at 16kHz. The
   *   underlying buffer is transferred to the worker (zero-copy), so the
   *   array is unusable to the caller afterwards.
   * @param {number} t0 - start bound (opaque, echoed back on 'result'/'dropped').
   * @param {number} t1 - end bound (opaque, echoed back on 'result'/'dropped').
   */
  transcribe(float32Audio, t0, t1) {
    if (!this._worker) {
      // No worker (init() not called, init failed, or destroyed): the job
      // can never run, so report it as dropped instead of vanishing.
      this.emit('dropped', { t0, t1 });
      return;
    }
    const id = ++this._idCounter;
    this._worker.postMessage(
      { type: 'transcribe', id, t0, t1, audio: float32Audio },
      [float32Audio.buffer]
    );
  }

  /**
   * Terminates the worker and releases the model from memory. Safe to call
   * multiple times or before init().
   */
  destroy() {
    if (this._rejectInit) {
      // Don't leave callers awaiting init() hanging forever.
      this._rejectInit(new Error('LocalASR: destroyed before init() completed'));
      this._rejectInit = null;
    }
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
    }
    this._isReady = false;
    this._initPromise = null;
  }
}
