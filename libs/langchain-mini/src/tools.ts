/**
 * tools.ts — Minimal tool/function calling support.
 *
 * Provides:
 *   - ToolDefinition     JSON Schema-based tool definition
 *   - BaseTool           abstract base class for tools
 *   - tool()             factory to create a BaseTool from a function
 *   - DynamicTool        simple string-in / string-out tool
 *   - renderToolsToOpenAI  convert tools to the OpenAI tools array format
 */

import { Runnable, type RunnableConfig } from "./runnable.js";
import { ToolMessage } from "./messages.js";

// ---------------------------------------------------------------------------
// JSON Schema subset used for tool parameters
// ---------------------------------------------------------------------------

/** A simplified JSON Schema type for describing tool parameters. */
export interface JsonSchema {
  type?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

// ---------------------------------------------------------------------------
// ToolDefinition — wire format
// ---------------------------------------------------------------------------

/**
 * Definition of a tool as passed to a chat model.
 * Compatible with OpenAI's function/tool format.
 */
export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: JsonSchema;
  };
}

// ---------------------------------------------------------------------------
// BaseTool
// ---------------------------------------------------------------------------

/** Configuration for creating a BaseTool. */
export interface BaseToolConfig {
  name: string;
  description: string;
  /** JSON Schema for the tool's input arguments. */
  schema: JsonSchema;
  /** Whether to return the tool output directly (skip further model calls). */
  returnDirect?: boolean;
}

/**
 * Abstract base class for all tools.
 *
 * Subclasses implement `_call(input, config?)` which receives the parsed
 * arguments object and returns a string result.
 *
 * Tools extend `Runnable<Record<string, unknown>, string>` so they can be
 * used inside pipelines.
 */
export abstract class BaseTool extends Runnable<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Record<string, any>,
  string
> {
  readonly name: string;
  readonly description: string;
  readonly schema: JsonSchema;
  readonly returnDirect: boolean;

  constructor(config: BaseToolConfig) {
    super();
    this.name = config.name;
    this.description = config.description;
    this.schema = config.schema;
    this.returnDirect = config.returnDirect ?? false;
  }

  abstract _call(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: Record<string, any>,
    config?: RunnableConfig
  ): Promise<string>;

  async invoke(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: Record<string, any>,
    config?: RunnableConfig
  ): Promise<string> {
    return this._call(input, config);
  }

  /**
   * Run the tool from a ToolCall (as found on an AIMessage).
   * Returns a ToolMessage suitable for feeding back into the model.
   */
  async run(
    toolCall: { id: string; name: string; args: Record<string, unknown> },
    config?: RunnableConfig
  ): Promise<ToolMessage> {
    const output = await this._call(toolCall.args, config);
    return new ToolMessage({
      content: output,
      tool_call_id: toolCall.id,
      name: this.name,
    });
  }

  /** Convert this tool to the OpenAI wire format. */
  toOpenAITool(): ToolDefinition {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: this.schema,
      },
    };
  }
}

// ---------------------------------------------------------------------------
// DynamicTool
// ---------------------------------------------------------------------------

/** A simple tool backed by an arbitrary function. */
export class DynamicTool extends BaseTool {
  private readonly fn: (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: Record<string, any>,
    config?: RunnableConfig
  ) => Promise<string> | string;

  constructor(
    config: BaseToolConfig & {
      fn: (
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        input: Record<string, any>,
        config?: RunnableConfig
      ) => Promise<string> | string;
    }
  ) {
    super(config);
    this.fn = config.fn;
  }

  async _call(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: Record<string, any>,
    config?: RunnableConfig
  ): Promise<string> {
    return this.fn(input, config);
  }
}

// ---------------------------------------------------------------------------
// tool() factory
// ---------------------------------------------------------------------------

/** Options for the `tool()` factory function. */
export interface ToolOptions {
  name: string;
  description: string;
  schema: JsonSchema;
  returnDirect?: boolean;
}

/**
 * Create a tool from a plain function.
 *
 * @example
 * const calculator = tool(
 *   async ({ expression }) => String(eval(expression)),
 *   {
 *     name: "calculator",
 *     description: "Evaluate a mathematical expression.",
 *     schema: {
 *       type: "object",
 *       properties: { expression: { type: "string" } },
 *       required: ["expression"],
 *     },
 *   }
 * );
 */
export function tool(
  fn: (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: Record<string, any>,
    config?: RunnableConfig
  ) => Promise<string> | string,
  options: ToolOptions
): DynamicTool {
  return new DynamicTool({ ...options, fn });
}

// ---------------------------------------------------------------------------
// Render tools for model APIs
// ---------------------------------------------------------------------------

/** Convert an array of BaseTool instances to OpenAI's tools format. */
export function renderToolsToOpenAI(tools: BaseTool[]): ToolDefinition[] {
  return tools.map((t) => t.toOpenAITool());
}

// ---------------------------------------------------------------------------
// ToolRegistry — look up tools by name for agent loops
// ---------------------------------------------------------------------------

/** A registry that maps tool names to BaseTool instances. */
export class ToolRegistry {
  private readonly tools: Map<string, BaseTool>;

  constructor(tools: BaseTool[]) {
    this.tools = new Map(tools.map((t) => [t.name, t]));
  }

  get(name: string): BaseTool | undefined {
    return this.tools.get(name);
  }

  list(): BaseTool[] {
    return [...this.tools.values()];
  }

  toOpenAITools(): ToolDefinition[] {
    return renderToolsToOpenAI(this.list());
  }

  /** Execute a ToolCall from an AIMessage and return a ToolMessage. */
  async execute(
    toolCall: { id: string; name: string; args: Record<string, unknown> },
    config?: RunnableConfig
  ): Promise<ToolMessage> {
    const t = this.get(toolCall.name);
    if (!t) {
      return new ToolMessage({
        content: `Tool "${toolCall.name}" not found.`,
        tool_call_id: toolCall.id,
      });
    }
    try {
      return await t.run(toolCall, config);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return new ToolMessage({
        content: `Error: ${msg}`,
        tool_call_id: toolCall.id,
      });
    }
  }
}
