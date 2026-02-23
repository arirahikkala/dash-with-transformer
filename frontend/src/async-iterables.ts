/**
 * Utilities for racing and merging async iterables and promises.
 */

/**
 * Race multiple async iterables, yielding values tagged with their source
 * index as soon as any source produces one.  Order across sources is
 * non-deterministic (whichever source's `.next()` settles first wins).
 */
export async function* raceAsyncIterables<T>(
  iterables: AsyncIterable<T>[],
): AsyncGenerator<{ value: T; index: number }> {
  const iterators = iterables.map((iter) => iter[Symbol.asyncIterator]());
  const active = new Map<
    number,
    Promise<{ result: IteratorResult<T>; index: number }>
  >();
  for (let i = 0; i < iterators.length; i++) {
    active.set(
      i,
      iterators[i].next().then((result) => ({ result, index: i })),
    );
  }
  while (active.size > 0) {
    const { result, index } = await Promise.race(active.values());
    if (result.done) {
      active.delete(index);
    } else {
      yield { value: result.value, index };
      active.set(
        index,
        iterators[index].next().then((r) => ({ result: r, index })),
      );
    }
  }
}

/**
 * Merge multiple async iterables, yielding values as soon as any source
 * produces one.  Order across sources is non-deterministic (whichever
 * source's `.next()` settles first wins).
 */
export async function* mergeAsyncIterables<T>(
  iterables: AsyncIterable<T>[],
): AsyncGenerator<T> {
  for await (const { value } of raceAsyncIterables(iterables)) {
    yield value;
  }
}

/**
 * Yield the results of an array of promises in resolution order
 * (whichever settles first is yielded first), rather than in array order.
 */
export async function* racePromises<T>(
  promises: Promise<T>[],
): AsyncGenerator<T> {
  const pending = new Map<number, Promise<{ value: T; index: number }>>();
  for (let i = 0; i < promises.length; i++) {
    pending.set(
      i,
      promises[i].then((value) => ({ value, index: i })),
    );
  }
  while (pending.size > 0) {
    const { value, index } = await Promise.race(pending.values());
    pending.delete(index);
    yield value;
  }
}
