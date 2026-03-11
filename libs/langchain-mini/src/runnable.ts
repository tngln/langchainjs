/**
 * runnable.ts — Minimal composable pipeline primitives.
 *
 * Provides:
 *   - Runnable<Input, Output>  base interface / abstract class
 *   - RunnableSequence         chain of runnables (pipe)
 *   - RunnableLambda           wrap a plain function as a Runnable
 *   - RunnablePassthrough      pass input through unchanged
 *   - RunnableMap              run multiple runnables in parallel over the same input
 *
 * Design goals:
 *   • Zero external dependencies.
 *   • Fully async — invoke/stream/batch.
 *   • Composable via pipe() / RunnableSequence.from().
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Options forwarded along the pipeline (abort signal, callbacks, etc.). */
export interface RunnableConfig {
  signal?: AbortSignal;
  /** Maximum number of concurrent calls in batch(). */
  maxConcurrency?: number;
  /** Arbitrary metadata carried through the pipeline. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata?: Record<string, any>;
  /** Run name for tracing. */
  runName?: string;
}

/** Anything that can act as a Runnable (instance OR plain function). */
export type RunnableLike<Input = unknown, Output = unknown> =
  | Runnable<Input, Output>
  | ((input: Input, config?: RunnableConfig) => Output | Promise<Output>);

// ---------------------------------------------------------------------------
// Abstract base
// ---------------------------------------------------------------------------

/**
 * Abstract base for all composable pipeline steps.
 *
 * Every Runnable must implement `invoke`.
 * `stream` defaults to wrapping `invoke` in a single-element async generator.
 * `batch` defaults to running `invoke` for each input in parallel.
 */
export abstract class Runnable<Input = unknown, Output = unknown> {
  abstract invoke(input: Input, config?: RunnableConfig): Promise<Output>;

  async *stream(
    input: Input,
    config?: RunnableConfig
  ): AsyncGenerator<Output> {
    yield await this.invoke(input, config);
  }

  async batch(
    inputs: Input[],
    config?: RunnableConfig
  ): Promise<Output[]> {
    const maxConcurrency = config?.maxConcurrency ?? inputs.length;
    const results: Output[] = new Array(inputs.length);

    // Process in batches of maxConcurrency
    for (let i = 0; i < inputs.length; i += maxConcurrency) {
      const slice = inputs.slice(i, i + maxConcurrency);
      const sliceResults = await Promise.all(
        slice.map((inp) => this.invoke(inp, config))
      );
      sliceResults.forEach((r, j) => {
        results[i + j] = r;
      });
    }

    return results;
  }

  /**
   * Pipe this runnable into the next, returning a RunnableSequence.
   */
  pipe<NewOutput>(
    next: RunnableLike<Output, NewOutput>
  ): RunnableSequence<Input, NewOutput> {
    return RunnableSequence.from([this, coerceToRunnable(next)]) as RunnableSequence<Input, NewOutput>;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Wrap a function or Runnable as a Runnable. */
function coerceToRunnable<Input, Output>(
  runnable: RunnableLike<Input, Output>
): Runnable<Input, Output> {
  if (runnable instanceof Runnable) return runnable;
  return new RunnableLambda<Input, Output>(runnable);
}

// ---------------------------------------------------------------------------
// RunnableSequence
// ---------------------------------------------------------------------------

/**
 * A linear sequence of Runnables.  The output of each step becomes the
 * input to the next.
 */
export class RunnableSequence<
  Input = unknown,
  Output = unknown,
> extends Runnable<Input, Output> {
  /** All steps in this sequence. */
  readonly steps: Runnable[];

  constructor(steps: Runnable[]) {
    super();
    if (steps.length < 2) {
      throw new Error("RunnableSequence requires at least 2 steps.");
    }
    this.steps = steps;
  }

  /** Build a RunnableSequence from an array of runnables / functions. */
  static from<First = unknown, Last = unknown>(
    steps: RunnableLike[]
  ): RunnableSequence<First, Last> {
    // Flatten nested sequences
    const flat: Runnable[] = [];
    for (const s of steps) {
      const r = coerceToRunnable(s);
      if (r instanceof RunnableSequence) {
        flat.push(...r.steps);
      } else {
        flat.push(r);
      }
    }
    return new RunnableSequence<First, Last>(flat);
  }

  async invoke(input: Input, config?: RunnableConfig): Promise<Output> {
    let current: unknown = input;
    for (const step of this.steps) {
      current = await step.invoke(current, config);
    }
    return current as Output;
  }

  async *stream(
    input: Input,
    config?: RunnableConfig
  ): AsyncGenerator<Output> {
    // Run all steps except the last synchronously, then stream the last.
    let current: unknown = input;
    for (let i = 0; i < this.steps.length - 1; i++) {
      current = await this.steps[i].invoke(current, config);
    }
    const last = this.steps[this.steps.length - 1];
    yield* last.stream(current, config) as AsyncGenerator<Output>;
  }
}

// ---------------------------------------------------------------------------
// RunnableLambda
// ---------------------------------------------------------------------------

/**
 * Wraps a plain function as a Runnable.
 *
 * If the function returns an AsyncGenerator it is treated as a streaming
 * source; otherwise the result is yielded in a single chunk.
 */
export class RunnableLambda<
  Input = unknown,
  Output = unknown,
> extends Runnable<Input, Output> {
  private readonly fn: (
    input: Input,
    config?: RunnableConfig
  ) => Output | Promise<Output>;

  constructor(
    fn: (input: Input, config?: RunnableConfig) => Output | Promise<Output>
  ) {
    super();
    this.fn = fn;
  }

  async invoke(input: Input, config?: RunnableConfig): Promise<Output> {
    const result = await this.fn(input, config);
    return result;
  }

  async *stream(
    input: Input,
    config?: RunnableConfig
  ): AsyncGenerator<Output> {
    const result = this.fn(input, config);
    // If the function returns an async iterable, stream from it directly.
    if (
      result !== null &&
      typeof result === "object" &&
      Symbol.asyncIterator in (result as object)
    ) {
      yield* result as unknown as AsyncIterable<Output>;
    } else {
      yield await result;
    }
  }
}

// ---------------------------------------------------------------------------
// RunnablePassthrough
// ---------------------------------------------------------------------------

/**
 * Returns its input unchanged.  Useful for building parallel maps or
 * injecting raw inputs into downstream steps.
 */
export class RunnablePassthrough<T = unknown> extends Runnable<T, T> {
  async invoke(input: T, _config?: RunnableConfig): Promise<T> {
    return input;
  }
}

// ---------------------------------------------------------------------------
// RunnableMap
// ---------------------------------------------------------------------------

/**
 * Runs multiple Runnables over the same input in parallel and returns
 * an object with the collected outputs keyed by the map key.
 *
 * @example
 * const map = RunnableMap.from({
 *   answer: chain,
 *   raw: new RunnablePassthrough(),
 * });
 * const result = await map.invoke(input);
 * // { answer: "...", raw: input }
 */
export class RunnableMap<
  Input = unknown,
  OutputRecord extends Record<string, unknown> = Record<string, unknown>,
> extends Runnable<Input, OutputRecord> {
  private readonly mappers: Record<string, Runnable<Input, unknown>>;

  constructor(mappers: Record<string, RunnableLike<Input, unknown>>) {
    super();
    this.mappers = Object.fromEntries(
      Object.entries(mappers).map(([k, v]) => [k, coerceToRunnable(v)])
    );
  }

  /** Build a RunnableMap from a plain object of runnables / functions. */
  static from<
    Input = unknown,
    OutputRecord extends Record<string, unknown> = Record<string, unknown>,
  >(
    mappers: { [K in keyof OutputRecord]: RunnableLike<Input, OutputRecord[K]> }
  ): RunnableMap<Input, OutputRecord> {
    return new RunnableMap<Input, OutputRecord>(
      mappers as Record<string, RunnableLike<Input, unknown>>
    );
  }

  async invoke(input: Input, config?: RunnableConfig): Promise<OutputRecord> {
    const entries = Object.entries(this.mappers);
    const values = await Promise.all(
      entries.map(([, r]) => r.invoke(input, config))
    );
    return Object.fromEntries(
      entries.map(([k], i) => [k, values[i]])
    ) as OutputRecord;
  }
}

// ---------------------------------------------------------------------------
// RunnableWithFallbacks
// ---------------------------------------------------------------------------

/**
 * Tries the primary Runnable; on failure, tries each fallback in order.
 */
export class RunnableWithFallbacks<
  Input = unknown,
  Output = unknown,
> extends Runnable<Input, Output> {
  private readonly primary: Runnable<Input, Output>;
  private readonly fallbacks: Runnable<Input, Output>[];

  constructor(
    primary: Runnable<Input, Output>,
    fallbacks: RunnableLike<Input, Output>[]
  ) {
    super();
    this.primary = primary;
    this.fallbacks = fallbacks.map(coerceToRunnable<Input, Output>);
  }

  async invoke(input: Input, config?: RunnableConfig): Promise<Output> {
    const chain = [this.primary, ...this.fallbacks];
    let lastError: unknown;
    for (const r of chain) {
      try {
        return await r.invoke(input, config);
      } catch (e) {
        lastError = e;
      }
    }
    throw lastError;
  }
}
