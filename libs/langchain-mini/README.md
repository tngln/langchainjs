# @langchain/mini

> A streamlined, minimal LangChain-compatible implementation in TypeScript.
> Zero runtime dependencies. ~2,000 lines of code.

## Why?

`@langchain/core` and the broader LangChain ecosystem are powerful but can
feel heavy for many use-cases. `@langchain/mini` covers the 90% use-case
with a tiny, auditable implementation you can understand end-to-end.

| Feature | Lines of code |
|---------|--------------|
| Messages (Human / AI / System / Tool) | ~270 |
| Runnable pipeline (sequence / lambda / map / fallbacks) | ~290 |
| Prompt templates (string + chat) | ~190 |
| Chat model base + FakeChatModel | ~320 |
| Output parsers | ~160 |
| Tools + ToolRegistry | ~230 |
| Memory / ConversationChain | ~200 |
| **Total** | **~1,660** |

## Installation

```bash
npm install @langchain/mini
```

## Quick Start

```typescript
import {
  ChatPromptTemplate,
  FakeChatModel,
  StringOutputParser,
  HumanMessage,
} from "@langchain/mini";

// 1. Build a prompt
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant called {name}."],
  ["human", "{question}"],
]);

// 2. Use any chat model (replace FakeChatModel with your provider)
const model = new FakeChatModel({ responses: ["The capital is Paris."] });

// 3. Compose a chain using pipe()
const chain = prompt
  .pipe(model)
  .pipe(new StringOutputParser());

// 4. Invoke
const answer = await chain.invoke({ name: "Mini", question: "What is the capital of France?" });
console.log(answer); // "The capital is Paris."
```

## Core Concepts

### Messages

```typescript
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from "@langchain/mini";

const human  = new HumanMessage("Hello!");
const ai     = new AIMessage("Hi there!");
const system = new SystemMessage("You are helpful.");
const tool   = new ToolMessage({ content: "42", tool_call_id: "call_1" });

// Shorthand for building messages from tuples
import { coerceMessageLike } from "@langchain/mini";
const msg = coerceMessageLike(["human", "Hello!"]);
```

### Runnable Pipeline

```typescript
import { RunnableLambda, RunnableSequence, RunnableMap, RunnablePassthrough } from "@langchain/mini";

const double  = new RunnableLambda<number, number>(async (x) => x * 2);
const addOne  = new RunnableLambda<number, number>(async (x) => x + 1);

// Pipe
const chain = double.pipe(addOne);
await chain.invoke(5); // 11

// Batch
await chain.batch([1, 2, 3]); // [3, 5, 7]

// Parallel map
const map = RunnableMap.from({
  doubled: double,
  original: new RunnablePassthrough(),
});
await map.invoke(4); // { doubled: 8, original: 4 }
```

### Prompt Templates

```typescript
import { PromptTemplate, ChatPromptTemplate } from "@langchain/mini";

// String prompt
const p = PromptTemplate.fromTemplate("Tell me a joke about {topic}.");
await p.invoke({ topic: "programmers" });

// Chat prompt
const chat = ChatPromptTemplate.fromMessages([
  ["system", "You are {name}."],
  ["human", "{question}"],
]);
const messages = await chat.invoke({ name: "Mini", question: "Hi!" });
```

### Implementing a Chat Model

```typescript
import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type ChatResult,
  type AnyMessage,
  AIMessage,
} from "@langchain/mini";

class MyChatModel extends BaseChatModel {
  async _generate(
    messages: AnyMessage[],
    options: BaseChatModelCallOptions
  ): Promise<ChatResult> {
    // Call your LLM API here...
    const text = "Hello from MyChatModel!";
    const message = new AIMessage(text);
    return { generations: [{ message, text }] };
  }
}
```

### Tools

```typescript
import { tool, ToolRegistry } from "@langchain/mini";

const search = tool(
  async ({ query }) => `Results for: ${query}`,
  {
    name: "search",
    description: "Search the web",
    schema: {
      type: "object",
      properties: { query: { type: "string", description: "Search query" } },
      required: ["query"],
    },
  }
);

// Bind tools to a model
const modelWithTools = model.bindTools([search]);

// Execute tool calls from an AI message
const registry = new ToolRegistry([search]);
const toolMessage = await registry.execute({ id: "call_1", name: "search", args: { query: "cats" } });
```

### Memory / Conversation History

```typescript
import { ConversationChain, BufferMemory } from "@langchain/mini";

const chain = new ConversationChain({
  model,
  memory: new BufferMemory({
    systemPrompt: "You are a helpful assistant.",
    maxPairs: 10, // keep last 10 exchanges
  }),
});

const r1 = await chain.invoke("Hello!");
const r2 = await chain.invoke("What did I just say?");
```

### Output Parsers

```typescript
import {
  StringOutputParser,
  JsonOutputParser,
  CommaSeparatedListOutputParser,
} from "@langchain/mini";

// Extract text
model.pipe(new StringOutputParser());

// Parse JSON (strips markdown fences automatically)
model.pipe(new JsonOutputParser<{ answer: string }>());

// Split comma-separated list
model.pipe(new CommaSeparatedListOutputParser());
```

## Streaming

```typescript
import { streamText } from "@langchain/mini";

// Stream tokens to the console
const fullText = await streamText(
  model.stream([new HumanMessage("Tell me a story")]),
  (token) => process.stdout.write(token)
);
```

## License

MIT
