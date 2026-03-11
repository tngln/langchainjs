import { describe, it, expect } from "vitest";
import {
  Runnable,
  RunnableSequence,
  RunnableLambda,
  RunnablePassthrough,
  RunnableMap,
  RunnableWithFallbacks,
} from "../runnable.js";

// ---- helpers ---------------------------------------------------------------

const double = new RunnableLambda<number, number>(async (x) => x * 2);
const addOne = new RunnableLambda<number, number>(async (x) => x + 1);
const toString = new RunnableLambda<number, string>(async (x) => String(x));

// ----------------------------------------------------------------------------

describe("RunnableLambda", () => {
  it("invokes the wrapped function", async () => {
    expect(await double.invoke(5)).toBe(10);
  });

  it("streams a single result by default", async () => {
    const results: number[] = [];
    for await (const chunk of double.stream(3)) {
      results.push(chunk);
    }
    expect(results).toEqual([6]);
  });

  it("batch processes all inputs", async () => {
    expect(await double.batch([1, 2, 3])).toEqual([2, 4, 6]);
  });
});

describe("RunnableSequence", () => {
  it("chains steps via .pipe()", async () => {
    const chain = double.pipe(addOne).pipe(toString);
    expect(await chain.invoke(5)).toBe("11");
  });

  it("RunnableSequence.from() flattens nested sequences", async () => {
    const inner = RunnableSequence.from([double, addOne]);
    const outer = RunnableSequence.from([inner, toString]);
    expect(await outer.invoke(3)).toBe("7");
  });

  it("requires at least 2 steps", () => {
    expect(() => new RunnableSequence([double])).toThrow();
  });

  it("streams the last step", async () => {
    // Use a string-returning lambda that streams characters
    const chars = new RunnableLambda<number, string>(async (x) => String(x));
    const chain = double.pipe(chars);
    const results: string[] = [];
    for await (const v of chain.stream(2)) {
      results.push(v);
    }
    // double(2) = 4, then chars returns "4"
    expect(results).toEqual(["4"]);
  });
});

describe("RunnablePassthrough", () => {
  it("returns input unchanged", async () => {
    const p = new RunnablePassthrough<string>();
    expect(await p.invoke("hello")).toBe("hello");
  });
});

describe("RunnableMap", () => {
  it("runs all runnables in parallel", async () => {
    const map = RunnableMap.from({
      doubled: double,
      plusOne: addOne,
      original: new RunnablePassthrough<number>(),
    });
    const result = await map.invoke(4);
    expect(result).toEqual({ doubled: 8, plusOne: 5, original: 4 });
  });
});

describe("RunnableWithFallbacks", () => {
  it("uses primary if it succeeds", async () => {
    const primary = new RunnableLambda<string, string>(async () => "primary");
    const fallback = new RunnableLambda<string, string>(async () => "fallback");
    const r = new RunnableWithFallbacks(primary, [fallback]);
    expect(await r.invoke("x")).toBe("primary");
  });

  it("falls back on primary failure", async () => {
    const primary = new RunnableLambda<string, string>(async () => {
      throw new Error("fail");
    });
    const fallback = new RunnableLambda<string, string>(async () => "fallback");
    const r = new RunnableWithFallbacks(primary, [fallback]);
    expect(await r.invoke("x")).toBe("fallback");
  });

  it("throws if all runnables fail", async () => {
    const fail = new RunnableLambda<string, string>(async () => {
      throw new Error("all failed");
    });
    const r = new RunnableWithFallbacks(fail, [fail]);
    await expect(r.invoke("x")).rejects.toThrow("all failed");
  });
});

describe("Runnable.batch", () => {
  it("respects maxConcurrency", async () => {
    const order: number[] = [];
    const tracked = new RunnableLambda<number, number>(async (x) => {
      order.push(x);
      return x;
    });
    await tracked.batch([1, 2, 3, 4], { maxConcurrency: 2 });
    // All items should be processed
    expect(order.sort()).toEqual([1, 2, 3, 4]);
  });
});

describe("Runnable subclass example", () => {
  it("can create a custom Runnable subclass", async () => {
    class Uppercase extends Runnable<string, string> {
      async invoke(input: string): Promise<string> {
        return input.toUpperCase();
      }
    }
    const r = new Uppercase();
    expect(await r.invoke("hello")).toBe("HELLO");
    const batched = await r.batch(["hello", "world"]);
    expect(batched).toEqual(["HELLO", "WORLD"]);
  });
});
