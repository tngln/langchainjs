import { describe, it, expect } from "vitest";
import { tool, DynamicTool, ToolRegistry, renderToolsToOpenAI } from "../tools.js";
import { ToolMessage } from "../messages.js";

const addTool = tool(
  async ({ a, b }) => String((a as number) + (b as number)),
  {
    name: "add",
    description: "Add two numbers",
    schema: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
    },
  }
);

describe("tool() factory", () => {
  it("creates a DynamicTool", () => {
    expect(addTool).toBeInstanceOf(DynamicTool);
    expect(addTool.name).toBe("add");
    expect(addTool.description).toBe("Add two numbers");
  });

  it("invokes correctly", async () => {
    const result = await addTool.invoke({ a: 3, b: 4 });
    expect(result).toBe("7");
  });

  it("converts to OpenAI tool definition", () => {
    const def = addTool.toOpenAITool();
    expect(def.type).toBe("function");
    expect(def.function.name).toBe("add");
    expect(def.function.parameters.properties).toBeDefined();
  });
});

describe("BaseTool.run()", () => {
  it("returns a ToolMessage", async () => {
    const result = await addTool.run({
      id: "call_1",
      name: "add",
      args: { a: 10, b: 20 },
    });
    expect(result).toBeInstanceOf(ToolMessage);
    expect(result.content).toBe("30");
    expect(result.tool_call_id).toBe("call_1");
  });
});

describe("ToolRegistry", () => {
  const registry = new ToolRegistry([addTool]);

  it("retrieves a tool by name", () => {
    expect(registry.get("add")).toBe(addTool);
    expect(registry.get("unknown")).toBeUndefined();
  });

  it("lists all tools", () => {
    expect(registry.list()).toHaveLength(1);
  });

  it("converts to OpenAI tools", () => {
    const tools = registry.toOpenAITools();
    expect(tools).toHaveLength(1);
    expect(tools[0].type).toBe("function");
  });

  it("executes a tool call successfully", async () => {
    const msg = await registry.execute({
      id: "call_x",
      name: "add",
      args: { a: 5, b: 6 },
    });
    expect(msg.content).toBe("11");
    expect(msg.tool_call_id).toBe("call_x");
  });

  it("returns error message for unknown tool", async () => {
    const msg = await registry.execute({
      id: "call_y",
      name: "unknown_tool",
      args: {},
    });
    expect(msg.content).toContain("not found");
  });

  it("returns error message when tool throws", async () => {
    const failTool = tool(
      async () => {
        throw new Error("oops");
      },
      {
        name: "fail",
        description: "always fails",
        schema: { type: "object", properties: {} },
      }
    );
    const reg = new ToolRegistry([failTool]);
    const msg = await reg.execute({ id: "z", name: "fail", args: {} });
    expect(msg.content).toContain("Error: oops");
  });
});

describe("renderToolsToOpenAI", () => {
  it("converts an array of tools to OpenAI format", () => {
    const defs = renderToolsToOpenAI([addTool]);
    expect(defs).toHaveLength(1);
    expect(defs[0].function.name).toBe("add");
  });
});
