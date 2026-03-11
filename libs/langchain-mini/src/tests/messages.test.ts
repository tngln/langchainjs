import { describe, it, expect } from "vitest";
import {
  HumanMessage,
  AIMessage,
  AIMessageChunk,
  SystemMessage,
  ToolMessage,
  coerceMessageLike,
} from "../messages.js";

describe("Messages", () => {
  it("HumanMessage stores string content", () => {
    const msg = new HumanMessage("Hello!");
    expect(msg.content).toBe("Hello!");
    expect(msg.type).toBe("human");
    expect(msg.getText()).toBe("Hello!");
  });

  it("AIMessage stores string content and empty tool_calls by default", () => {
    const msg = new AIMessage("Hi there!");
    expect(msg.content).toBe("Hi there!");
    expect(msg.type).toBe("ai");
    expect(msg.tool_calls).toEqual([]);
    expect(msg.invalid_tool_calls).toEqual([]);
  });

  it("AIMessage stores tool_calls", () => {
    const msg = new AIMessage({
      content: "",
      tool_calls: [{ id: "1", name: "search", args: { q: "LangChain" } }],
    });
    expect(msg.tool_calls).toHaveLength(1);
    expect(msg.tool_calls[0].name).toBe("search");
  });

  it("SystemMessage stores content", () => {
    const msg = new SystemMessage("You are a helpful assistant.");
    expect(msg.type).toBe("system");
    expect(msg.getText()).toBe("You are a helpful assistant.");
  });

  it("ToolMessage stores content and tool_call_id", () => {
    const msg = new ToolMessage({
      content: "42",
      tool_call_id: "call_abc",
    });
    expect(msg.type).toBe("tool");
    expect(msg.tool_call_id).toBe("call_abc");
    expect(msg.getText()).toBe("42");
  });

  it("getText joins multimodal content blocks", () => {
    const msg = new HumanMessage([
      { type: "text", text: "Hello " },
      { type: "text", text: "world" },
    ]);
    expect(msg.getText()).toBe("Hello world");
  });

  it("AIMessageChunk can be concatenated", () => {
    const chunk1 = new AIMessageChunk("Hel");
    const chunk2 = new AIMessageChunk("lo!");
    const combined = chunk1.concat(chunk2);
    expect(combined.getText()).toBe("Hello!");
  });

  it("AIMessageChunk concatenates tool_call_chunks by index", () => {
    const c1 = new AIMessageChunk({
      content: "",
      tool_call_chunks: [{ index: 0, id: "call1", name: "search", argsRaw: '{"q":' }],
    });
    const c2 = new AIMessageChunk({
      content: "",
      tool_call_chunks: [{ index: 0, argsRaw: '"cats"}' }],
    });
    const combined = c1.concat(c2);
    expect(combined.tool_call_chunks).toHaveLength(1);
    expect(combined.tool_call_chunks[0].argsRaw).toBe('{"q":"cats"}');
  });

  it("AIMessageChunk merges usage_metadata", () => {
    const c1 = new AIMessageChunk({
      content: "a",
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    });
    const c2 = new AIMessageChunk({
      content: "b",
      usage_metadata: { input_tokens: 0, output_tokens: 3, total_tokens: 3 },
    });
    const combined = c1.concat(c2);
    expect(combined.usage_metadata?.output_tokens).toBe(8);
    expect(combined.usage_metadata?.total_tokens).toBe(18);
  });

  describe("coerceMessageLike", () => {
    it("passes through an existing message", () => {
      const msg = new HumanMessage("hi");
      expect(coerceMessageLike(msg)).toBe(msg);
    });

    it("wraps a string as HumanMessage", () => {
      const msg = coerceMessageLike("hello");
      expect(msg).toBeInstanceOf(HumanMessage);
      expect(msg.getText()).toBe("hello");
    });

    it("handles tuple [role, content]", () => {
      const msg = coerceMessageLike(["ai", "I am an AI"]);
      expect(msg).toBeInstanceOf(AIMessage);
      expect(msg.getText()).toBe("I am an AI");
    });

    it("throws for unknown role", () => {
      // @ts-expect-error - testing invalid role
      expect(() => coerceMessageLike(["unknown", "x"])).toThrow();
    });
  });

  describe("isInstance", () => {
    it("HumanMessage.isInstance", () => {
      expect(HumanMessage.isInstance(new HumanMessage("hi"))).toBe(true);
      expect(HumanMessage.isInstance(new AIMessage("hi"))).toBe(false);
    });

    it("AIMessage.isInstance", () => {
      expect(AIMessage.isInstance(new AIMessage("hi"))).toBe(true);
      expect(AIMessage.isInstance("not a message")).toBe(false);
    });
  });
});
