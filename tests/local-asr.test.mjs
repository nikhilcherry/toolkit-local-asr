// Run with: node --test
//
// local-asr.js is main-thread postMessage plumbing around a Web Worker.
// Node has no Worker global, so these tests install a stub that captures
// construction and messages, and lets each test play the worker's side of
// the conversation by invoking `onmessage` directly. The real inference
// path lives in asr-worker.js (not loadable outside a browser); the
// queue-drop policy it uses is covered by transcribe-queue.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';

class StubWorker {
  static instances = [];

  constructor(url, opts) {
    this.url = url;
    this.opts = opts;
    this.posted = [];
    this.transfers = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
    StubWorker.instances.push(this);
  }

  postMessage(msg, transfer) {
    this.posted.push(msg);
    this.transfers.push(transfer);
  }

  terminate() {
    this.terminated = true;
  }

  // test helper: simulate a message coming FROM the worker
  receive(data) {
    this.onmessage({ data });
  }
}

globalThis.Worker = StubWorker;
const { LocalASR } = await import('../local-asr.js');

function lastWorker() {
  return StubWorker.instances[StubWorker.instances.length - 1];
}

test('init() posts the init message and resolves on ready', async () => {
  const asr = new LocalASR({ model: 'test-model', language: null, device: 'wasm', maxQueue: 3 });
  const initPromise = asr.init();
  const worker = lastWorker();

  assert.deepEqual(worker.posted[0], {
    type: 'init', model: 'test-model', language: null, device: 'wasm', maxQueue: 3,
  });
  assert.equal(asr.isReady, false);

  worker.receive({ type: 'ready' });
  await initPromise;
  assert.equal(asr.isReady, true);
});

test('init() is idempotent: repeat calls share one worker', async () => {
  const asr = new LocalASR();
  const before = StubWorker.instances.length;
  const p1 = asr.init();
  const p2 = asr.init();
  assert.equal(p1, p2);
  assert.equal(StubWorker.instances.length, before + 1);
  lastWorker().receive({ type: 'ready' });
  await p1;
});

test('a failed init() terminates the worker and is retryable', async () => {
  const asr = new LocalASR();
  const p1 = asr.init();
  const worker1 = lastWorker();
  worker1.receive({ type: 'init-error', error: 'model download failed' });

  await assert.rejects(p1, /model download failed/);
  assert.equal(worker1.terminated, true, 'dead worker must be terminated');

  // retry spins up a fresh worker instead of replaying the cached failure
  const p2 = asr.init();
  const worker2 = lastWorker();
  assert.notEqual(worker1, worker2);
  worker2.receive({ type: 'ready' });
  await p2;
  assert.equal(asr.isReady, true);
});

test('destroy() during a pending init() rejects the init promise', async () => {
  const asr = new LocalASR();
  const initPromise = asr.init();
  asr.destroy();
  await assert.rejects(initPromise, /destroyed before init/);
  assert.equal(lastWorker().terminated, true);
});

test('transcribe() posts the job and transfers the audio buffer', async () => {
  const asr = new LocalASR();
  const initPromise = asr.init();
  const worker = lastWorker();
  worker.receive({ type: 'ready' });
  await initPromise;

  const audio = new Float32Array([0.1, 0.2, 0.3]);
  asr.transcribe(audio, 100, 200);

  const job = worker.posted[1];
  assert.equal(job.type, 'transcribe');
  assert.equal(job.t0, 100);
  assert.equal(job.t1, 200);
  assert.deepEqual(worker.transfers[1], [audio.buffer]);
});

test('transcribe() with no worker emits dropped instead of vanishing', () => {
  const asr = new LocalASR();
  const dropped = [];
  asr.on('dropped', (d) => dropped.push(d));

  asr.transcribe(new Float32Array([0.1]), 5, 10);

  assert.deepEqual(dropped, [{ t0: 5, t1: 10 }]);
});

test('result/dropped/status worker messages re-emit as events', async () => {
  const asr = new LocalASR();
  const initPromise = asr.init();
  const worker = lastWorker();
  worker.receive({ type: 'ready' });
  await initPromise;

  const events = [];
  asr.on('result', (e) => events.push(['result', e]));
  asr.on('dropped', (e) => events.push(['dropped', e]));
  asr.on('status', (e) => events.push(['status', e]));

  worker.receive({ type: 'result', text: 'hello', t0: 0, t1: 4000, ms: 350 });
  worker.receive({ type: 'dropped', t0: 4000, t1: 8000 });
  worker.receive({ type: 'error', error: 'boom' });

  assert.deepEqual(events, [
    ['result', { text: 'hello', t0: 0, t1: 4000, ms: 350 }],
    ['dropped', { t0: 4000, t1: 8000 }],
    ['status', 'transcription error: boom'],
  ]);
});

test('a worker crash after init surfaces as a status event', async () => {
  const asr = new LocalASR();
  const initPromise = asr.init();
  const worker = lastWorker();
  worker.receive({ type: 'ready' });
  await initPromise;

  const statuses = [];
  asr.on('status', (s) => statuses.push(s));
  worker.onerror({ message: 'worker exploded' });

  assert.equal(statuses.length, 1);
  assert.match(statuses[0], /worker exploded/);
  assert.equal(asr.isReady, true, 'a post-init crash must not clobber ready state');
});

test('destroy() is safe to call twice and before init()', () => {
  const asr = new LocalASR();
  assert.doesNotThrow(() => asr.destroy());
  assert.doesNotThrow(() => asr.destroy());
});
