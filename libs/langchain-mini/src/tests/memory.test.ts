import { describe, it, expect } from "vitest";
import {
  ChatMessageHistory,
  BufferMemory,
  ConversationChain,
  RunnableWithMessageHistory,
} from "../memory.js";
import { FakeChatModel, type BaseChatModelCallOptions, type ChatResult } from "../chat_models.js";
import { HumanMessage, SystemMessage, type AnyMessage } from "../messages.js";

describe("ChatMessageHistory", () => {
  it("stores and retrieves messages", () => {
    const history = new ChatMessageHistory();
    history.addUserMessage("Hello");
    history.addAIMessage("Hi there!");
    const messages = history.getMessages();
    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(HumanMessage);
    expect(messages[0].getText()).toBe("Hello");
  });

  it("clears all messages", () => {
    const history = new ChatMessageHistory();
    history.addUserMessage("Hi");
    history.clear();
    expect(history.getMessages()).toHaveLength(0);
    expect(history.length).toBe(0);
  });

  it("returns a copy of messages (no mutation)", () => {
    const history = new ChatMessageHistory();
    history.addUserMessage("A");
    const msgs1 = history.getMessages();
    msgs1.push(new HumanMessage("B"));
    expect(history.getMessages()).toHaveLength(1);
  });
});

describe("BufferMemory", () => {
  it("builds messages with system prompt and history", () => {
    const memory = new BufferMemory({ systemPrompt: "You are helpful." });
    memory.saveContext("Hello", "Hi!");
    const messages = memory.buildMessages("How are you?");

    expect(messages[0]).toBeInstanceOf(SystemMessage);
    expect(messages[0].getText()).toBe("You are helpful.");
    expect(messages).toHaveLength(4); // system + 2 history + 1 new human
  });

  it("respects maxPairs limit", () => {
    const memory = new BufferMemory({ maxPairs: 1 });
    memory.saveContext("Turn 1", "Response 1");
    memory.saveContext("Turn 2", "Response 2");
    const messages = memory.buildMessages("Turn 3");
    // Only last 1 pair (2 messages) + new human
    expect(messages).toHaveLength(3);
    expect(messages[0].getText()).toBe("Turn 2");
  });

  it("builds messages without system prompt", () => {
    const memory = new BufferMemory();
    const messages = memory.buildMessages("Hello");
    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(HumanMessage);
  });
});

describe("ConversationChain", () => {
  it("maintains conversation history across turns", async () => {
    const model = new FakeChatModel({ responses: ["Hello!", "I am Mini."] });
    const chain = new ConversationChain({
      model,
      memory: new BufferMemory({ systemPrompt: "You are Mini." }),
    });

    const r1 = await chain.invoke("Hello");
    expect(r1).toBe("Hello!");

    const r2 = await chain.invoke("Who are you?");
    expect(r2).toBe("I am Mini.");

    // Check that history is accumulated
    const history = chain.memory.chatHistory.getMessages();
    expect(history).toHaveLength(4); // 2 human + 2 ai
  });

  it("uses default memory if not provided", async () => {
    const model = new FakeChatModel({ responses: ["ok"] });
    const chain = new ConversationChain({ model });
    await chain.invoke("hi");
    expect(chain.memory.chatHistory.length).toBe(2);
  });
});

describe("RunnableWithMessageHistory", () => {
  it("loads and saves history per session", async () => {
    const sessions = new Map<string, ChatMessageHistory>();
    const getHistory = (id: string) => {
      if (!sessions.has(id)) sessions.set(id, new ChatMessageHistory());
      return sessions.get(id)!;
    };

    const model = new FakeChatModel({ responses: ["Hello A!", "Hello B!"] });
    const withHistory = new RunnableWithMessageHistory({
      runnable: model,
      getSessionHistory: getHistory,
    });

    await withHistory.invoke([new HumanMessage("Hi")], {
      metadata: { sessionId: "session-a" },
    });
    await withHistory.invoke([new HumanMessage("Hi")], {
      metadata: { sessionId: "session-b" },
    });

    // Each session should have its own independent history
    expect(sessions.get("session-a")!.length).toBe(2);
    expect(sessions.get("session-b")!.length).toBe(2);
  });

  it("prepends system prompt when provided", async () => {
    const received: AnyMessage[] = [];

    class SpyModel extends FakeChatModel {
      async _generate(
        messages: AnyMessage[],
        options: BaseChatModelCallOptions
      ): Promise<ChatResult> {
        received.push(...messages.filter((m) => m instanceof SystemMessage));
        return super._generate(messages, options);
      }
    }

    const model = new SpyModel({ responses: ["ok"] });
    const withHistory = new RunnableWithMessageHistory({
      runnable: model,
      getSessionHistory: () => new ChatMessageHistory(),
      systemPrompt: "You are a test assistant.",
    });

    await withHistory.invoke([new HumanMessage("Hello")], {});
    expect(received[0]).toBeInstanceOf(SystemMessage);
    expect(received[0].getText()).toBe("You are a test assistant.");
  });
});
