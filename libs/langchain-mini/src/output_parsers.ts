/**
 * output_parsers.ts — Minimal output parser implementations.
 *
 * Provides:
 *   - StringOutputParser      extract the text content from an AIMessage
 *   - JsonOutputParser        parse AIMessage text as JSON
 *   - CommaSeparatedParser    split a comma-delimited list into an array
 *   - RegexParser             extract named groups from a regex match
 *
 * All parsers extend `Runnable<AIMessage, T>` so they compose naturally
 * at the end of a pipeline (model.pipe(parser)).
 */

import { Runnable, type RunnableConfig } from "./runnable.js";
import { AIMessage, AIMessageChunk } from "./messages.js";

// ---------------------------------------------------------------------------
// StringOutputParser
// ---------------------------------------------------------------------------

/**
 * Extracts the plain-text content from an `AIMessage` or `AIMessageChunk`.
 *
 * @example
 * const chain = model.pipe(new StringOutputParser());
 * const text = await chain.invoke(messages);
 */
export class StringOutputParser extends Runnable<
  AIMessage | AIMessageChunk,
  string
> {
  async invoke(
    message: AIMessage | AIMessageChunk,
    _config?: RunnableConfig
  ): Promise<string> {
    return message.getText();
  }

  async *stream(
    message: AIMessage | AIMessageChunk,
    _config?: RunnableConfig
  ): AsyncGenerator<string> {
    yield message.getText();
  }
}

// ---------------------------------------------------------------------------
// JsonOutputParser
// ---------------------------------------------------------------------------

/**
 * Parses AIMessage content as JSON, stripping Markdown code fences first.
 *
 * @example
 * const chain = model.pipe(new JsonOutputParser<{ answer: string }>());
 * const data = await chain.invoke(messages);
 */
export class JsonOutputParser<T = unknown> extends Runnable<
  AIMessage | AIMessageChunk,
  T
> {
  async invoke(
    message: AIMessage | AIMessageChunk,
    _config?: RunnableConfig
  ): Promise<T> {
    return parseJson<T>(message.getText());
  }
}

/** Strip markdown code fences and parse JSON. */
function parseJson<T>(text: string): T {
  const cleaned = text
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```\s*$/, "")
    .trim();
  return JSON.parse(cleaned) as T;
}

// ---------------------------------------------------------------------------
// CommaSeparatedListOutputParser
// ---------------------------------------------------------------------------

/**
 * Splits the model's text response on commas and trims whitespace.
 *
 * @example
 * const chain = model.pipe(new CommaSeparatedListOutputParser());
 * const items = await chain.invoke(messages); // ["item1", "item2", ...]
 */
export class CommaSeparatedListOutputParser extends Runnable<
  AIMessage | AIMessageChunk,
  string[]
> {
  async invoke(
    message: AIMessage | AIMessageChunk,
    _config?: RunnableConfig
  ): Promise<string[]> {
    return message
      .getText()
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }
}

// ---------------------------------------------------------------------------
// RegexParser
// ---------------------------------------------------------------------------

/**
 * Extract named capture groups from the model's text output using a regex.
 *
 * @example
 * const parser = new RegexParser(/Action: (?<action>\w+)\nInput: (?<input>.+)/s, ["action", "input"]);
 * const result = await chain.pipe(parser).invoke(messages);
 * // { action: "Search", input: "LangChain" }
 */
export class RegexParser extends Runnable<
  AIMessage | AIMessageChunk,
  Record<string, string>
> {
  readonly regex: RegExp;
  readonly outputKeys: string[];

  constructor(regex: RegExp, outputKeys: string[]) {
    super();
    this.regex = regex;
    this.outputKeys = outputKeys;
  }

  async invoke(
    message: AIMessage | AIMessageChunk,
    _config?: RunnableConfig
  ): Promise<Record<string, string>> {
    const text = message.getText();
    const match = text.match(this.regex);
    if (!match?.groups) {
      throw new Error(
        `Regex did not match the model output.\nOutput: ${text}\nRegex: ${this.regex}`
      );
    }
    const out: Record<string, string> = {};
    for (const key of this.outputKeys) {
      out[key] = match.groups[key] ?? "";
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// StructuredOutputParser (field-based)
// ---------------------------------------------------------------------------

/**
 * Parses a model's JSON output and validates that each expected key is
 * present.  Returns a typed record.
 */
export class StructuredOutputParser<
  T extends Record<string, unknown> = Record<string, unknown>,
> extends Runnable<AIMessage | AIMessageChunk, T> {
  readonly keys: string[];

  constructor(keys: string[]) {
    super();
    this.keys = keys;
  }

  /** Build from a plain key→description map. */
  static fromNames<T extends Record<string, unknown>>(
    names: Record<keyof T, string>
  ): StructuredOutputParser<T> {
    return new StructuredOutputParser<T>(Object.keys(names));
  }

  async invoke(
    message: AIMessage | AIMessageChunk,
    _config?: RunnableConfig
  ): Promise<T> {
    const parsed = parseJson<Record<string, unknown>>(message.getText());
    for (const key of this.keys) {
      if (!(key in parsed)) {
        throw new Error(`Missing key in structured output: "${key}"`);
      }
    }
    return parsed as T;
  }
}
