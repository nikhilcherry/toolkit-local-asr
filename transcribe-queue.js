/**
 * TranscribeQueue — the "oldest queued job is dropped once at capacity"
 * policy described in README's Queue-drop policy section, pulled out of
 * asr-worker.js so it's testable without a Worker context or the
 * transformers.js CDN import. Holds only jobs that haven't started running
 * yet: asr-worker.js removes a job from this queue (via shift()) the
 * moment it starts inference, so "currently running" is never in here to
 * begin with -- that's what makes "never evict the running job" true by
 * construction rather than a case this class has to check for.
 */
export class TranscribeQueue {
  /**
   * @param {number} maxQueue - max queued (not-yet-running) jobs before
   *   the oldest is evicted to make room for a new one.
   */
  constructor(maxQueue) {
    /** @type {number} */
    this.maxQueue = maxQueue;
    /** @type {any[]} */
    this._items = [];
  }

  /** @returns {number} */
  get length() {
    return this._items.length;
  }

  /**
   * Adds a job. If already at capacity, evicts and returns the oldest
   * queued job first to make room; returns null if nothing was evicted.
   * @param {any} job
   * @returns {any|null} the evicted job, or null
   */
  push(job) {
    let evicted = null;
    if (this._items.length >= this.maxQueue) {
      evicted = this._items.shift();
    }
    this._items.push(job);
    return evicted;
  }

  /**
   * Removes and returns the oldest queued job, or undefined if empty.
   * @returns {any|undefined}
   */
  shift() {
    return this._items.shift();
  }
}
