// Run with: node --test
//
// Everything else in this tool -- init(), transcribe(), the worker's
// pipeline() calls -- needs a Worker context and the transformers.js CDN
// import, neither available in plain Node. TranscribeQueue is the one
// piece of real logic (the "drop the oldest queued job at capacity"
// policy README documents) with no such dependency.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TranscribeQueue } from '../transcribe-queue.js';

test('push below capacity does not evict, FIFO order preserved', () => {
  const q = new TranscribeQueue(3);
  assert.equal(q.push('a'), null);
  assert.equal(q.push('b'), null);
  assert.equal(q.length, 2);
  assert.equal(q.shift(), 'a');
  assert.equal(q.shift(), 'b');
});

test('push at exact capacity evicts the oldest queued job', () => {
  const q = new TranscribeQueue(2);
  q.push('a');
  q.push('b');
  const evicted = q.push('c'); // at capacity (2) -- must evict 'a'
  assert.equal(evicted, 'a');
  assert.equal(q.length, 2);
  assert.equal(q.shift(), 'b');
  assert.equal(q.shift(), 'c');
});

test('a burst well past capacity evicts one at a time, newest jobs survive', () => {
  const q = new TranscribeQueue(2);
  const evictions = [];
  for (const job of ['a', 'b', 'c', 'd', 'e']) {
    const evicted = q.push(job);
    if (evicted !== null) evictions.push(evicted);
  }
  assert.deepEqual(evictions, ['a', 'b', 'c']);
  assert.equal(q.length, 2);
  assert.equal(q.shift(), 'd');
  assert.equal(q.shift(), 'e');
});

test('shift() on an empty queue returns undefined, does not throw', () => {
  const q = new TranscribeQueue(2);
  assert.equal(q.shift(), undefined);
});

test('maxQueue of 0: eviction check runs before insertion, so a 0-capacity queue transiently holds 1 item', () => {
  // push() checks length >= maxQueue BEFORE inserting the new job, so the
  // very first push into a 0-capacity queue has nothing to evict yet (the
  // queue is still empty at check time) -- eviction only bites on the
  // *next* push. This is the existing asr-worker.js algorithm's actual
  // behavior at this edge, preserved as-is rather than changed here.
  const q = new TranscribeQueue(0);
  // shift() on the still-empty backing array returns undefined here, not
  // the push()-level "nothing evicted" null seen when maxQueue > 0.
  assert.equal(q.push('a'), undefined);
  assert.equal(q.length, 1);
  assert.equal(q.push('b'), 'a');
  assert.equal(q.length, 1);
});

test('maxQueue can be tuned live (mirrors asr-worker.js re-init behavior)', () => {
  const q = new TranscribeQueue(5);
  q.push('a');
  q.push('b');
  q.maxQueue = 1; // e.g. LocalASR re-initialized with a smaller maxQueue
  const evicted = q.push('c'); // length (2) >= new maxQueue (1) -> evict oldest
  assert.equal(evicted, 'a');
  assert.equal(q.length, 2); // shrinking maxQueue doesn't retroactively trim existing items
});

test('a job that starts running is removed from the queue and can never be evicted', () => {
  // Mirrors asr-worker.js's processQueue(): the running job is taken out
  // via shift() before any further push() calls can see it, so "never
  // evict the currently running job" holds by construction.
  const q = new TranscribeQueue(1);
  q.push('running');
  const running = q.shift(); // simulates processQueue() starting inference
  assert.equal(running, 'running');
  assert.equal(q.length, 0);

  q.push('queued-1');
  const evicted = q.push('queued-2'); // capacity 1 -- evicts queued-1, not "running"
  assert.equal(evicted, 'queued-1');
});
