import { describe, it, expect } from "vitest";
import {
  FakeChatModel,
  BaseChatModel,
  type BaseChatModelCallOptions,
  type ChatResult,
  streamToFull,
  streamText,
} from "../chat_models.js";
import { HumanMessage, AIMessage, AIMessageChunk, type AnyMessage } from "../messages.js";
import {
  StringOutputParser,
  JsonOutputParser,
  CommaSeparatedListOutputParser,
} from "../output_parsers.js";
import { ChatPromptTemplate } from "../prompts.js";
import { type RunnableConfig } from "../runnable.js";

describe("FakeChatModel", () => {
  it("returns the configured response", async () => {
    const model = new FakeChatModel({ responses: ["Hello!"] });
    const result = await model.invoke([new HumanMessage("Hi")]);
    expect(result).toBeInstanceOf(AIMessage);
    expect(result.getText()).toBe("Hello!");
  });

  it("cycles through multiple responses", async () => {
    const model = new FakeChatModel({
      responses: ["First", "Second", "Third"],
    });
    const r1 = await model.invoke([new HumanMessage("a")]);
    const r2 = await model.invoke([new HumanMessage("b")]);
    const r3 = await model.invoke([new HumanMessage("c")]);
    const r4 = await model.invoke([new HumanMessage("d")]); // wraps
    expect(r1.getText()).toBe("First");
    expect(r2.getText()).toBe("Second");
    expect(r3.getText()).toBe("Third");
    expect(r4.getText()).toBe("First");
  });

  it("streams character by character", async () => {
    const model = new FakeChatModel({ responses: ["abc"] });
    const chunks: string[] = [];
    for await (const chunk of model.stream([new HumanMessage("x")])) {
      chunks.push(chunk.getText());
    }
    expect(chunks).toEqual(["a", "b", "c"]);
  });

  it("batch processes inputs", async () => {
    const model = new FakeChatModel({
      responses: ["one", "two", "three"],
    });
    const results = await model.batch([
      [new HumanMessage("a")],
      [new HumanMessage("b")],
      [new HumanMessage("c")],
    ]);
    expect(results.map((r) => r.getText())).toEqual(["one", "two", "three"]);
  });

  it("works with AIMessage responses", async () => {
    const msg = new AIMessage({
      content: "",
      tool_calls: [{ id: "1", name: "calc", args: { x: 1 } }],
    });
    const model = new FakeChatModel({ responses: [msg] });
    const result = await model.invoke([new HumanMessage("use tool")]);
    expect(result.tool_calls).toHaveLength(1);
    expect(result.tool_calls[0].name).toBe("calc");
  });
});

describe("streamToFull", () => {
  it("aggregates chunks", async () => {
    const model = new FakeChatModel({ responses: ["hello world"] });
    const stream = model.stream([new HumanMessage("x")]);
    const full = await streamToFull(stream);
    expect(full.getText()).toBe("hello world");
  });
});

describe("streamText", () => {
  it("returns full text and calls callback per token", async () => {
    const model = new FakeChatModel({ responses: ["abc"] });
    const tokens: string[] = [];
    const result = await streamText(
      model.stream([new HumanMessage("x")]),
      (t) => tokens.push(t)
    );
    expect(result).toBe("abc");
    expect(tokens).toEqual(["a", "b", "c"]);
  });
});

describe("BaseChatModel pipeline integration", () => {
  it("model.pipe(StringOutputParser) returns a string chain", async () => {
    const model = new FakeChatModel({ responses: ["Hello, world!"] });
    const chain = model.pipe(new StringOutputParser());
    const text = await chain.invoke([new HumanMessage("Hi")]);
    expect(typeof text).toBe("string");
    expect(text).toBe("Hello, world!");
  });

  it("PromptTemplate | model | parser chain works end-to-end", async () => {
    const model = new FakeChatModel({ responses: ["Meow"] });
    const chain = ChatPromptTemplate.fromMessages([
      ["human", "Tell me something about {animal}"],
    ])
      .pipe(model)
      .pipe(new StringOutputParser());

    const result = await chain.invoke({ animal: "cats" });
    expect(result).toBe("Meow");
  });

  it("model | JsonOutputParser chain works end-to-end", async () => {
    const model = new FakeChatModel({
      responses: ['{"answer": 42, "unit": "meters"}'],
    });
    const chain = model.pipe(new JsonOutputParser<{ answer: number; unit: string }>());
    const result = await chain.invoke([new HumanMessage("test")]);
    expect(result.answer).toBe(42);
    expect(result.unit).toBe("meters");
  });

  it("model | JsonOutputParser handles markdown fences", async () => {
    const model = new FakeChatModel({
      responses: ["```json\n{\"ok\": true}\n```"],
    });
    const chain = model.pipe(new JsonOutputParser<{ ok: boolean }>());
    const result = await chain.invoke([new HumanMessage("test")]);
    expect(result.ok).toBe(true);
  });

  it("CommaSeparatedListOutputParser splits response", async () => {
    const model = new FakeChatModel({ responses: ["cats, dogs, fish"] });
    const chain = model.pipe(new CommaSeparatedListOutputParser());
    const result = await chain.invoke([new HumanMessage("list animals")]);
    expect(result).toEqual(["cats", "dogs", "fish"]);
  });
});

describe("withStructuredOutput", () => {
  it("parses JSON from model response", async () => {
    const model = new FakeChatModel({
      responses: ['{"name": "Mini", "version": 1}'],
    });
    const structured = model.withStructuredOutput<{ name: string; version: number }>(
      {}
    );
    const result = await structured.invoke([new HumanMessage("describe yourself")]);
    // result is either T or { raw, parsed } depending on includeRaw option
    const parsed = result as { name: string; version: number };
    expect(parsed.name).toBe("Mini");
    expect(parsed.version).toBe(1);
  });

  it("includeRaw returns both raw and parsed", async () => {
    const model = new FakeChatModel({ responses: ['{"x": 99}'] });
    const structured = model.withStructuredOutput<{ x: number }>(
      {},
      { includeRaw: true }
    );
    const result = await structured.invoke([new HumanMessage("test")]);
    // result is { raw: AIMessage | AIMessageChunk; parsed: T }
    const withRaw = result as { raw: AIMessage | AIMessageChunk; parsed: { x: number } };
    expect(withRaw.parsed.x).toBe(99);
    expect(withRaw.raw).toBeInstanceOf(AIMessage);
  });
});

describe("bindTools", () => {
  it("creates a new model instance with tools bound", () => {
    const model = new FakeChatModel({ responses: ["ok"] });
    const toolDef = {
      type: "function" as const,
      function: {
        name: "search",
        description: "Search the web",
        parameters: { type: "object", properties: {} },
      },
    };
    const modelWithTools = model.bindTools([toolDef]);
    // boundTools should be set on the new instance, not the original
    expect((modelWithTools as unknown as { boundTools: unknown[] }).boundTools).toHaveLength(1);
    expect((model as unknown as { boundTools: unknown }).boundTools).toBeUndefined();
  });
});

describe("Custom BaseChatModel subclass", () => {
  class EchoModel extends BaseChatModel {
    async _generate(
      messages: AnyMessage[],
      _options: BaseChatModelCallOptions,
      _config?: RunnableConfig
    ): Promise<ChatResult> {
      const lastMsg = messages[messages.length - 1];
      const echo = new AIMessage(`ECHO: ${lastMsg.getText()}`);
      return { generations: [{ message: echo, text: echo.getText() }] };
    }
  }

  it("subclass invokes correctly", async () => {
    const model = new EchoModel({ model: "echo-1" });
    const result = await model.invoke([new HumanMessage("hello")]);
    expect(result.getText()).toBe("ECHO: hello");
  });

  it("subclass inherits streaming from _generate", async () => {
    const model = new EchoModel({ model: "echo-1" });
    const chunks: string[] = [];
    for await (const chunk of model.stream([new HumanMessage("test")])) {
      chunks.push(chunk.getText());
    }
    expect(chunks.join("")).toBe("ECHO: test");
  });
});
