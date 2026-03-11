import { describe, it, expect } from "vitest";
import { PromptTemplate, ChatPromptTemplate, systemHumanPrompt } from "../prompts.js";
import { HumanMessage, SystemMessage, AIMessage } from "../messages.js";

describe("PromptTemplate", () => {
  it("interpolates variables", async () => {
    const prompt = PromptTemplate.fromTemplate("Tell me a joke about {topic}.");
    const result = await prompt.invoke({ topic: "cats" });
    expect(result).toBe("Tell me a joke about cats.");
  });

  it("extracts inputVariables", () => {
    const prompt = PromptTemplate.fromTemplate("{a} and {b}");
    expect(prompt.inputVariables).toEqual(expect.arrayContaining(["a", "b"]));
  });

  it("throws on missing variable", async () => {
    const prompt = PromptTemplate.fromTemplate("Hello {name}!");
    // @ts-expect-error - intentionally passing wrong type
    await expect(prompt.invoke({})).rejects.toThrow("Missing template variable");
  });

  it("handles multiple occurrences of the same variable", async () => {
    const prompt = PromptTemplate.fromTemplate("{x} + {x} = {y}");
    const result = await prompt.invoke({ x: "2", y: "4" });
    expect(result).toBe("2 + 2 = 4");
  });
});

describe("ChatPromptTemplate", () => {
  it("formats messages from tuples", async () => {
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are {name}."],
      ["human", "{question}"],
    ]);
    const messages = await prompt.invoke({ name: "Mini", question: "Hello!" });
    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(SystemMessage);
    expect(messages[0].getText()).toBe("You are Mini.");
    expect(messages[1]).toBeInstanceOf(HumanMessage);
    expect(messages[1].getText()).toBe("Hello!");
  });

  it("collects inputVariables from all messages", () => {
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "I am {assistant}."],
      ["human", "Tell me about {topic}."],
    ]);
    expect(prompt.inputVariables).toEqual(
      expect.arrayContaining(["assistant", "topic"])
    );
  });

  it("supports object form message templates", async () => {
    const prompt = ChatPromptTemplate.fromMessages([
      { role: "system", content: "You are helpful." },
      { role: "human", content: "Hi there!" },
    ]);
    const messages = await prompt.invoke({});
    expect(messages).toHaveLength(2);
    expect(messages[1].getText()).toBe("Hi there!");
  });

  it("handles ai role in messages", async () => {
    const prompt = ChatPromptTemplate.fromMessages([
      ["human", "Hello!"],
      ["ai", "Hello! How can I help you?"],
      ["human", "{followup}"],
    ]);
    const messages = await prompt.invoke({ followup: "What is 2+2?" });
    expect(messages[1]).toBeInstanceOf(AIMessage);
    expect(messages[2].getText()).toBe("What is 2+2?");
  });

  it("format() is synchronous", () => {
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are {name}."],
    ]);
    const messages = prompt.format({ name: "Sync" });
    expect(messages[0].getText()).toBe("You are Sync.");
  });
});

describe("systemHumanPrompt", () => {
  it("creates a system+human prompt", async () => {
    const prompt = systemHumanPrompt("You are {assistant_name}.");
    const messages = await prompt.invoke({
      assistant_name: "Mini",
      input: "Hello!",
    });
    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(SystemMessage);
    expect(messages[0].getText()).toBe("You are Mini.");
    expect(messages[1].getText()).toBe("Hello!");
  });

  it("accepts a custom human template", async () => {
    const prompt = systemHumanPrompt("Be concise.", "Answer: {answer}");
    const messages = await prompt.invoke({ answer: "42" });
    expect(messages[1].getText()).toBe("Answer: 42");
  });
});
