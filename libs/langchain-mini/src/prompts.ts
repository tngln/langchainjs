/**
 * prompts.ts — Minimal prompt template support.
 *
 * Provides:
 *   - PromptTemplate        simple {variable} string interpolation → string
 *   - ChatPromptTemplate    array of message templates → AnyMessage[]
 *
 * Both extend Runnable so they compose naturally with chains.
 */

import {
  AnyMessage,
  HumanMessage,
  AIMessage,
  SystemMessage,
  MessageContent,
  MessageRole,
} from "./messages.js";
import { Runnable, type RunnableConfig } from "./runnable.js";

// ---------------------------------------------------------------------------
// Template helpers
// ---------------------------------------------------------------------------

/** Template variable names in a template string. */
type TemplateVariables<T extends string> =
  T extends `${string}{${infer Var}}${infer Rest}`
    ? Var | TemplateVariables<Rest>
    : never;

/**
 * Interpolate `{variable}` placeholders in a template string.
 * Throws if any placeholder is missing from `values`.
 */
function interpolate(
  template: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  values: Record<string, any>
): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    if (!(key in values)) {
      throw new Error(`Missing template variable: "${key}"`);
    }
    return String(values[key]);
  });
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyRecord = Record<string, any>;

/**
 * A simple string prompt template using `{variable}` placeholders.
 *
 * @example
 * const prompt = PromptTemplate.fromTemplate("Tell me a joke about {topic}.");
 * const text = await prompt.invoke({ topic: "cats" });
 * // "Tell me a joke about cats."
 */
export class PromptTemplate<
  TTemplate extends string = string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TInput extends Record<TemplateVariables<TTemplate>, any> = Record<
    TemplateVariables<TTemplate>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    any
  >,
> extends Runnable<TInput, string> {
  readonly template: string;
  readonly inputVariables: string[];

  constructor(template: string) {
    super();
    this.template = template;
    // Extract variable names from {variable} placeholders
    this.inputVariables = [
      ...new Set([...template.matchAll(/\{(\w+)\}/g)].map((m) => m[1])),
    ];
  }

  /** Convenience constructor from a template string. */
  static fromTemplate<T extends string>(
    template: T
  ): PromptTemplate<
    T,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Record<TemplateVariables<T>, any>
  > {
    return new PromptTemplate<
      T,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      Record<TemplateVariables<T>, any>
    >(template);
  }

  async invoke(input: TInput, _config?: RunnableConfig): Promise<string> {
    return interpolate(this.template, input as AnyRecord);
  }
}

// ---------------------------------------------------------------------------
// MessageTemplate
// ---------------------------------------------------------------------------

/** A single message template inside a ChatPromptTemplate. */
export interface MessageTemplate {
  role: MessageRole;
  /** A template string with {variable} placeholders, or a fixed MessageContent. */
  content: string | MessageContent;
}

/** Shorthand tuple form: [role, template_string]. */
export type MessageTemplateTuple = [MessageRole, string];

/** Accepts either the object form or the shorthand tuple. */
export type MessageTemplateLike = MessageTemplate | MessageTemplateTuple;

function normalizeTemplate(t: MessageTemplateLike): MessageTemplate {
  if (Array.isArray(t)) {
    return { role: t[0], content: t[1] };
  }
  return t;
}

// ---------------------------------------------------------------------------
// ChatPromptTemplate
// ---------------------------------------------------------------------------

/**
 * A prompt template that formats a list of chat messages.
 * Supports {variable} placeholders in message content strings.
 *
 * @example
 * const prompt = ChatPromptTemplate.fromMessages([
 *   ["system", "You are a helpful assistant called {name}."],
 *   ["human", "{question}"],
 * ]);
 * const messages = await prompt.invoke({ name: "Mini", question: "Hello!" });
 */
export class ChatPromptTemplate<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TInput extends AnyRecord = AnyRecord,
> extends Runnable<TInput, AnyMessage[]> {
  readonly messages: MessageTemplate[];
  readonly inputVariables: string[];

  constructor(messages: MessageTemplateLike[]) {
    super();
    this.messages = messages.map(normalizeTemplate);
    // Collect all template variables across all message content strings
    const vars = new Set<string>();
    for (const msg of this.messages) {
      if (typeof msg.content === "string") {
        for (const m of msg.content.matchAll(/\{(\w+)\}/g)) {
          vars.add(m[1]);
        }
      }
    }
    this.inputVariables = [...vars];
  }

  /** Convenience constructor from an array of template tuples or objects. */
  static fromMessages<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TInput extends AnyRecord = AnyRecord,
  >(
    messages: MessageTemplateLike[]
  ): ChatPromptTemplate<TInput> {
    return new ChatPromptTemplate<TInput>(messages);
  }

  async invoke(input: TInput, _config?: RunnableConfig): Promise<AnyMessage[]> {
    return this.format(input);
  }

  /** Synchronous format — useful when you don't need the Runnable interface. */
  format(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    values: Record<string, any>
  ): AnyMessage[] {
    return this.messages.map((msg) => {
      const content =
        typeof msg.content === "string"
          ? interpolate(msg.content, values)
          : msg.content;
      return buildMessage(msg.role, content);
    });
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildMessage(role: MessageRole, content: MessageContent): AnyMessage {
  switch (role) {
    case "human":
      return new HumanMessage(content);
    case "ai":
      return new AIMessage(content);
    case "system":
      return new SystemMessage(content);
    default:
      // Fallback — treat unknown roles as human
      return new HumanMessage(content);
  }
}

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

/**
 * One-liner factory for the most common pattern: system + human.
 *
 * @example
 * const prompt = systemHumanPrompt("You are {assistant_name}.");
 * const messages = await prompt.invoke({ assistant_name: "Mini", input: "Hi!" });
 */
export function systemHumanPrompt(
  systemTemplate: string,
  humanTemplate = "{input}"
): ChatPromptTemplate {
  return ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["human", humanTemplate],
  ]);
}
