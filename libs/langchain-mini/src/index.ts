/**
 * @langchain/mini — A streamlined, minimal LangChain-compatible implementation.
 *
 * Covers the 90% use-case in ~2000 lines of TypeScript with zero runtime
 * dependencies outside Node's built-ins.
 *
 * Re-exports everything from the individual modules so consumers can do:
 *
 *   import { HumanMessage, ChatPromptTemplate, StringOutputParser } from "@langchain/mini";
 */

// Messages
export {
  // Types
  type ContentBlock,
  type MessageContent,
  type MessageRole,
  type ToolCall,
  type ToolCallChunk,
  type InvalidToolCall,
  type UsageMetadata,
  type AnyMessage,
  type MessageTuple,
  type BaseMessageFields,
  type ToolMessageFields,
  type AIMessageFields,
  // Classes
  BaseMessage,
  HumanMessage,
  AIMessage,
  AIMessageChunk,
  SystemMessage,
  ToolMessage,
  // Helpers
  coerceMessageLike,
} from "./messages.js";

// Runnable
export {
  // Types
  type RunnableConfig,
  type RunnableLike,
  // Classes
  Runnable,
  RunnableSequence,
  RunnableLambda,
  RunnablePassthrough,
  RunnableMap,
  RunnableWithFallbacks,
} from "./runnable.js";

// Prompts
export {
  // Types
  type MessageTemplate,
  type MessageTemplateTuple,
  type MessageTemplateLike,
  // Classes
  PromptTemplate,
  ChatPromptTemplate,
  // Helpers
  systemHumanPrompt,
} from "./prompts.js";

// Chat Models
export {
  // Types
  type ChatGeneration,
  type ChatResult,
  type BaseChatModelCallOptions,
  type BaseChatModelConfig,
  type FakeChatModelOptions,
  // Classes
  BaseChatModel,
  FakeChatModel,
  // Helpers
  streamToFull,
  streamText,
} from "./chat_models.js";

// Output Parsers
export {
  StringOutputParser,
  JsonOutputParser,
  CommaSeparatedListOutputParser,
  RegexParser,
  StructuredOutputParser,
} from "./output_parsers.js";

// Tools
export {
  // Types
  type JsonSchema,
  type ToolDefinition,
  type BaseToolConfig,
  type ToolOptions,
  // Classes
  BaseTool,
  DynamicTool,
  ToolRegistry,
  // Helpers
  tool,
  renderToolsToOpenAI,
} from "./tools.js";

// Memory / History
export {
  ChatMessageHistory,
  BufferMemory,
  type BufferMemoryConfig,
  ConversationChain,
  RunnableWithMessageHistory,
} from "./memory.js";
