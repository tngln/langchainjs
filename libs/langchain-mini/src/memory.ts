/**
 * memory.ts — Minimal chat history / memory support.
 *
 * Provides:
 *   - ChatMessageHistory      in-memory conversation history
 *   - BufferMemory            wraps a history for use with chains
 *   - ConversationChain       model + memory in one composable Runnable
 */

import {
  AnyMessage,
  HumanMessage,
  AIMessage,
  AIMessageChunk,
  SystemMessage,
} from "./messages.js";
import { Runnable, type RunnableConfig } from "./runnable.js";
import { BaseChatModel } from "./chat_models.js";

// ---------------------------------------------------------------------------
// ChatMessageHistory
// ---------------------------------------------------------------------------

/**
 * An in-memory store of conversation messages.
 * Can be swapped for a persistent backend by subclassing.
 */
export class ChatMessageHistory {
  private messages: AnyMessage[] = [];

  /** Add a human message. */
  addUserMessage(text: string): void {
    this.messages.push(new HumanMessage(text));
  }

  /** Add an AI message. */
  addAIMessage(text: string): void {
    this.messages.push(new AIMessage(text));
  }

  /** Add any message directly. */
  addMessage(message: AnyMessage): void {
    this.messages.push(message);
  }

  /** Retrieve all stored messages. */
  getMessages(): AnyMessage[] {
    return [...this.messages];
  }

  /** Remove all messages. */
  clear(): void {
    this.messages = [];
  }

  /** Current message count. */
  get length(): number {
    return this.messages.length;
  }
}

// ---------------------------------------------------------------------------
// BufferMemory
// ---------------------------------------------------------------------------

/** Configuration for BufferMemory. */
export interface BufferMemoryConfig {
  /** System prompt to prepend to every conversation (optional). */
  systemPrompt?: string;
  /** Optionally pass an existing ChatMessageHistory instance. */
  chatHistory?: ChatMessageHistory;
  /**
   * Maximum number of recent message *pairs* (human + ai) to keep.
   * Older pairs are dropped.  Default: unlimited.
   */
  maxPairs?: number;
}

/**
 * Wraps a `ChatMessageHistory` and exposes helpers for building message
 * arrays suitable for passing to a chat model.
 */
export class BufferMemory {
  readonly chatHistory: ChatMessageHistory;
  private readonly systemPrompt?: string;
  private readonly maxPairs?: number;

  constructor(config: BufferMemoryConfig = {}) {
    this.chatHistory = config.chatHistory ?? new ChatMessageHistory();
    this.systemPrompt = config.systemPrompt;
    this.maxPairs = config.maxPairs;
  }

  /** Build the current message list to send to the model. */
  buildMessages(userInput: string): AnyMessage[] {
    const messages: AnyMessage[] = [];

    if (this.systemPrompt) {
      messages.push(new SystemMessage(this.systemPrompt));
    }

    let history = this.chatHistory.getMessages();
    if (this.maxPairs != null) {
      // Keep only the last N pairs (2*N messages)
      history = history.slice(-this.maxPairs * 2);
    }

    messages.push(...history, new HumanMessage(userInput));
    return messages;
  }

  /** Save the last round-trip (user input + model response). */
  saveContext(userInput: string, aiOutput: string): void {
    this.chatHistory.addUserMessage(userInput);
    this.chatHistory.addAIMessage(aiOutput);
  }

  /** Clear all history. */
  clear(): void {
    this.chatHistory.clear();
  }
}

// ---------------------------------------------------------------------------
// ConversationChain
// ---------------------------------------------------------------------------

/**
 * A simple Runnable that combines a chat model with a `BufferMemory` to
 * maintain conversation context across multiple turns.
 *
 * @example
 * const chain = new ConversationChain({
 *   model,
 *   memory: new BufferMemory({ systemPrompt: "You are a helpful assistant." }),
 * });
 *
 * const r1 = await chain.invoke("Hello!");
 * const r2 = await chain.invoke("What did I just say?");
 */
export class ConversationChain extends Runnable<string, string> {
  private readonly model: BaseChatModel;
  readonly memory: BufferMemory;

  constructor(config: { model: BaseChatModel; memory?: BufferMemory }) {
    super();
    this.model = config.model;
    this.memory = config.memory ?? new BufferMemory();
  }

  async invoke(userInput: string, config?: RunnableConfig): Promise<string> {
    const messages = this.memory.buildMessages(userInput);
    const response = await this.model.invoke(messages, config);
    const text = response.getText();
    this.memory.saveContext(userInput, text);
    return text;
  }
}

// ---------------------------------------------------------------------------
// RunnableWithMessageHistory
// ---------------------------------------------------------------------------

/**
 * Wraps any `Runnable<AnyMessage[], AIMessage>` (typically a chat model or
 * chain) and automatically loads / saves conversation history.
 *
 * @example
 * const withHistory = new RunnableWithMessageHistory({
 *   runnable: model,
 *   getSessionHistory: (sessionId) => mySessionStore.get(sessionId),
 * });
 * const response = await withHistory.invoke(
 *   [new HumanMessage("Hello!")],
 *   { metadata: { sessionId: "user-123" } }
 * );
 */
export class RunnableWithMessageHistory extends Runnable<
  AnyMessage[],
  AIMessage | AIMessageChunk
> {
  private readonly runnable: Runnable<AnyMessage[], AIMessage | AIMessageChunk>;
  private readonly getHistory: (sessionId: string) => ChatMessageHistory;
  private readonly systemPrompt?: string;

  constructor(config: {
    runnable: Runnable<AnyMessage[], AIMessage | AIMessageChunk>;
    getSessionHistory: (sessionId: string) => ChatMessageHistory;
    systemPrompt?: string;
  }) {
    super();
    this.runnable = config.runnable;
    this.getHistory = config.getSessionHistory;
    this.systemPrompt = config.systemPrompt;
  }

  async invoke(
    inputMessages: AnyMessage[],
    config?: RunnableConfig
  ): Promise<AIMessage | AIMessageChunk> {
    const sessionId = (config?.metadata?.["sessionId"] as string) ?? "default";
    const history = this.getHistory(sessionId);

    const messages: AnyMessage[] = [];
    if (this.systemPrompt) {
      messages.push(new SystemMessage(this.systemPrompt));
    }
    messages.push(...history.getMessages(), ...inputMessages);

    const response = await this.runnable.invoke(messages, config);

    // Persist the round-trip
    for (const msg of inputMessages) {
      history.addMessage(msg);
    }
    // Store as AIMessage for history purposes
    const historyMsg = new AIMessage(response.getText());
    history.addMessage(historyMsg);

    return response;
  }
}
