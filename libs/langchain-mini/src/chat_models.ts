/**
 * chat_models.ts — Minimal chat model base class.
 *
 * Provides:
 *   - BaseChatModel       abstract base with invoke / stream / batch / bindTools
 *   - withStructuredOutput  wrap a model to always return parsed JSON
 *   - ChatResult / ChatGeneration
 *
 * Concrete implementations (OpenAI, Anthropic, …) subclass BaseChatModel and
 * implement _generate() and optionally _stream().
 */

import {
  AnyMessage,
  AIMessage,
  AIMessageChunk,
} from "./messages.js";
import { Runnable, type RunnableConfig } from "./runnable.js";
import { type BaseTool, type ToolDefinition } from "./tools.js";

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/** A single generation from the model. */
export interface ChatGeneration {
  message: AIMessage;
  /** Convenience text accessor. */
  text: string;
}

/** The full response from a chat model call. */
export interface ChatResult {
  generations: ChatGeneration[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  llmOutput?: Record<string, any>;
}

// ---------------------------------------------------------------------------
// Call options
// ---------------------------------------------------------------------------

/** Options that can be passed per-call to a chat model. */
export interface BaseChatModelCallOptions {
  /** Maximum number of tokens to generate. */
  maxTokens?: number;
  /** Sampling temperature. */
  temperature?: number;
  /** Top-P nucleus sampling. */
  topP?: number;
  /** Stop sequences. */
  stop?: string[];
  /** Tools the model can call. */
  tools?: (BaseTool | ToolDefinition)[];
  /** How the model should use tools ("auto" | "required" | "none" | specific). */
  toolChoice?: "auto" | "required" | "none" | { name: string };
  /** Abort signal. */
  signal?: AbortSignal;
}

// ---------------------------------------------------------------------------
// BaseChatModel
// ---------------------------------------------------------------------------

/** Configuration passed to the BaseChatModel constructor. */
export interface BaseChatModelConfig {
  /** Model identifier (e.g. "gpt-4o", "claude-3-5-sonnet-20241022"). */
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  stop?: string[];
}

/**
 * Abstract base for chat models.
 *
 * Subclasses must implement `_generate()`.
 * Optionally override `_stream()` for native streaming support.
 */
export abstract class BaseChatModel<
  CallOptions extends BaseChatModelCallOptions = BaseChatModelCallOptions,
> extends Runnable<AnyMessage[], AIMessage | AIMessageChunk> {
  readonly model: string;
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly topP?: number;
  readonly stop?: string[];

  /** Tools bound to this model instance (via bindTools). */
  protected boundTools?: (BaseTool | ToolDefinition)[];
  protected boundToolChoice?: CallOptions["toolChoice"];

  constructor(config: BaseChatModelConfig = {}) {
    super();
    this.model = config.model ?? "unknown";
    this.temperature = config.temperature;
    this.maxTokens = config.maxTokens;
    this.topP = config.topP;
    this.stop = config.stop;
  }

  // ---------------------------------------------------------------------------
  // Abstract / override points
  // ---------------------------------------------------------------------------

  /**
   * Subclasses implement this to call the underlying model API.
   * Should throw on API errors.
   */
  abstract _generate(
    messages: AnyMessage[],
    options: CallOptions,
    config?: RunnableConfig
  ): Promise<ChatResult>;

  /**
   * Subclasses may override to provide native streaming.
   * Default implementation calls `_generate` and wraps result in a single chunk.
   */
  async *_stream(
    messages: AnyMessage[],
    options: CallOptions,
    _config?: RunnableConfig
  ): AsyncGenerator<AIMessageChunk> {
    const result = await this._generate(messages, options, _config);
    const gen = result.generations[0];
    if (gen) {
      yield new AIMessageChunk({
        content: gen.message.content,
        tool_calls: gen.message.tool_calls,
        usage_metadata: gen.message.usage_metadata,
        response_metadata: gen.message.response_metadata,
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Runnable interface
  // ---------------------------------------------------------------------------

  async invoke(
    messages: AnyMessage[],
    config?: RunnableConfig
  ): Promise<AIMessage> {
    const callOptions = this._buildCallOptions(config);
    const result = await this._generate(messages, callOptions, config);
    return result.generations[0]?.message ?? new AIMessage("");
  }

  async *stream(
    messages: AnyMessage[],
    config?: RunnableConfig
  ): AsyncGenerator<AIMessageChunk> {
    const callOptions = this._buildCallOptions(config);
    yield* this._stream(messages, callOptions, config);
  }

  /** Stream and aggregate all chunks into a final AIMessageChunk. */
  async streamFull(
    messages: AnyMessage[],
    config?: RunnableConfig
  ): Promise<AIMessageChunk> {
    let result: AIMessageChunk | undefined;
    for await (const chunk of this.stream(messages, config)) {
      result = result ? result.concat(chunk) : chunk;
    }
    return result ?? new AIMessageChunk("");
  }

  // ---------------------------------------------------------------------------
  // Tool binding
  // ---------------------------------------------------------------------------

  /**
   * Return a new model instance with the given tools pre-bound.
   * Subsequent calls will include these tools automatically.
   */
  bindTools(
    tools: (BaseTool | ToolDefinition)[],
    options?: { toolChoice?: CallOptions["toolChoice"] }
  ): this {
    const clone = Object.create(
      Object.getPrototypeOf(this),
      Object.getOwnPropertyDescriptors(this)
    ) as this;
    clone.boundTools = tools;
    clone.boundToolChoice = options?.toolChoice;
    return clone;
  }

  // ---------------------------------------------------------------------------
  // Structured output
  // ---------------------------------------------------------------------------

  /**
   * Wrap the model to always return a parsed JSON object matching a schema.
   *
   * The returned Runnable will:
   *   1. Call the model.
   *   2. Parse the text output as JSON.
   *   3. Return the parsed object.
   */
  withStructuredOutput<T = unknown>(
    _schema: object,
    options?: { includeRaw?: boolean }
  ): Runnable<AnyMessage[], T> | Runnable<AnyMessage[], { raw: AIMessage | AIMessageChunk; parsed: T }> {
    const model = this;
    if (options?.includeRaw) {
      return new StructuredOutputRunnableWithRaw<T>(model);
    }
    return new StructuredOutputRunnable<T>(model);
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private _buildCallOptions(config?: RunnableConfig): CallOptions {
    return {
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      topP: this.topP,
      stop: this.stop,
      tools: this.boundTools,
      toolChoice: this.boundToolChoice,
      signal: config?.signal,
    } as CallOptions;
  }
}

// ---------------------------------------------------------------------------
// StructuredOutputRunnable
// ---------------------------------------------------------------------------

class StructuredOutputRunnable<T> extends Runnable<AnyMessage[], T> {
  constructor(private readonly model: BaseChatModel) {
    super();
  }

  async invoke(messages: AnyMessage[], config?: RunnableConfig): Promise<T> {
    const response = await this.model.invoke(messages, config);
    const text = response.getText();
    // Strip markdown code fences if present
    const cleaned = text
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```\s*$/, "")
      .trim();
    return JSON.parse(cleaned) as T;
  }
}

class StructuredOutputRunnableWithRaw<T> extends Runnable<
  AnyMessage[],
  { raw: AIMessage | AIMessageChunk; parsed: T }
> {
  constructor(private readonly model: BaseChatModel) {
    super();
  }

  async invoke(
    messages: AnyMessage[],
    config?: RunnableConfig
  ): Promise<{ raw: AIMessage | AIMessageChunk; parsed: T }> {
    const raw = await this.model.invoke(messages, config);
    const text = raw.getText();
    const cleaned = text
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```\s*$/, "")
      .trim();
    return { raw, parsed: JSON.parse(cleaned) as T };
  }
}

// ---------------------------------------------------------------------------
// FakeChatModel — deterministic stub for tests
// ---------------------------------------------------------------------------

/** Options for FakeChatModel. */
export interface FakeChatModelOptions {
  /**
   * Responses to cycle through.  Each call to `invoke` returns the next
   * response (wrapping around if exhausted).
   */
  responses: (string | AIMessage)[];
  /** Simulated per-token streaming delay in milliseconds. Default 0. */
  streamDelay?: number;
}

/**
 * A deterministic fake chat model for unit tests.
 * Does not make any network calls.
 *
 * @example
 * const model = new FakeChatModel({ responses: ["Hello!"] });
 * const msg = await model.invoke([new HumanMessage("Hi")]);
 * // msg.content === "Hello!"
 */
export class FakeChatModel extends BaseChatModel {
  private readonly responses: AIMessage[];
  private callCount = 0;
  private readonly streamDelay: number;

  constructor(options: FakeChatModelOptions) {
    super({ model: "fake" });
    this.responses = options.responses.map((r) =>
      typeof r === "string" ? new AIMessage(r) : r
    );
    this.streamDelay = options.streamDelay ?? 0;
  }

  async _generate(
    _messages: AnyMessage[],
    _options: BaseChatModelCallOptions
  ): Promise<ChatResult> {
    const msg = this.responses[this.callCount % this.responses.length];
    this.callCount++;
    return {
      generations: [{ message: msg, text: msg.getText() }],
    };
  }

  async *_stream(
    messages: AnyMessage[],
    options: BaseChatModelCallOptions
  ): AsyncGenerator<AIMessageChunk> {
    const result = await this._generate(messages, options);
    const text = result.generations[0]?.message.getText() ?? "";
    for (const char of text) {
      if (this.streamDelay > 0) {
        await new Promise((res) => setTimeout(res, this.streamDelay));
      }
      yield new AIMessageChunk(char);
    }
  }
}

// ---------------------------------------------------------------------------
// Utility: aggregate streaming chunks
// ---------------------------------------------------------------------------

/**
 * Collect all chunks from a streaming call and concatenate them into a
 * single AIMessageChunk.
 */
export async function streamToFull(
  stream: AsyncIterable<AIMessageChunk>
): Promise<AIMessageChunk> {
  let result: AIMessageChunk | undefined;
  for await (const chunk of stream) {
    result = result ? result.concat(chunk) : chunk;
  }
  return result ?? new AIMessageChunk("");
}

/**
 * Collect streaming text chunks and call the callback on each.
 * Returns the full concatenated text.
 */
export async function streamText(
  stream: AsyncIterable<AIMessageChunk>,
  onChunk?: (token: string) => void
): Promise<string> {
  let full = "";
  for await (const chunk of stream) {
    const text =
      typeof chunk.content === "string"
        ? chunk.content
        : chunk
            .getText();
    full += text;
    onChunk?.(text);
  }
  return full;
}
