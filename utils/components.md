# Data Processing Components for RAG

In Retrieval-Augmented Generation (RAG) and other LLM applications, preparing your text data is a critical first step. This typically involves three main processes: **Tokenization**, **Chunking**, and **Embedding**.

## 1. Tokenization

**What is it?**
Tokenization is the process of breaking down a stream of text into smaller meaningful units called "tokens". A token can be a word, a part of a word (subword), or even a single character.

**How does it work?**
Language models don't read text like humans do; they process numbers. Tokenization maps text to numerical IDs that the model's neural network can process.
There are different types of tokenizers (e.g., Byte-Pair Encoding (BPE), WordPiece, SentencePiece). Models like OpenAI's GPT use a specific tokenizer (like `tiktoken`) where an average token represents ~3/4 of a typical English word.

**Do we use Tokens in RAG?**
Yes and No.
- **Directly manually managing tokens?** Usually **No**. We rarely write code to explicitly tokenize text into lists of integers before chunking when using frameworks like LangChain, unless we are doing very specific token-based length counting to fit exact context windows.
- **Why not?** Frameworks abstract this away. Instead of counting tokens directly, we often chunk based on characters or words for simplicity. However, *advanced* chunkers (like LangChain's `TokenTextSplitter`) do use tokenization under the hood to ensure chunks precisely fit within a model's token limit.

## 2. Chunking (Text Splitting)

**What is it?**
Chunking is the process of taking large pieces of text (like a full document) and breaking them down into smaller, manageable pieces (chunks).

**Why do we need it?**
1. **Context Window Limits:** LLMs have a strict limit on how much text they can process at once (e.g., 8k, 16k, or 128k tokens). You can't fit an entire book into the prompt.
2. **Retrieval Accuracy:** Embedding an entire document results in a diluted embedding vector. Embedding smaller, focused chunks ensures higher accuracy when retrieving relevant context for the RAG pipeline.

**Types of Chunking in LangChain:**
| Splitter Type | Best For | Key Advantage |
|---|---|---|
| **RecursiveCharacter** | General text, articles | Most popular; splits by paragraphs, then sentences, then words to keep context. |
| **TokenTextSplitter** | LLM pipelines | Splits strictly by token count (e.g., via tiktoken) to ensure you never hit model limits. |
| **MarkdownHeaderText** | .md files | Understands headers (#, ##) and keeps content with its title. |
| **CodeSplitter** | Python, JS, etc. | Understands code syntax (functions, classes) and avoids splitting in the middle of a logic block. |
| **NLP-based (spaCy/NLTK)**| Complex grammar | Uses real linguistic models to ensure sentences are never cut in half. |
| **Semantic Chunker** | Semantic search | Groups text based on the meaning of the content using embeddings rather than just character count. |

## 3. Embedding

**What is it?**
Embedding is the process of translating text chunks into high-dimensional numerical vectors (lists of floats).

**How does it work?**
Embedding models (like OpenAI's `text-embedding-3-small` or HuggingFace models) process a chunk of text and output a vector that captures its semantic meaning. Texts that are similar in meaning will have vectors that are closer together in this high-dimensional space.

**Types of Embeddings in LangChain:**
- **OpenAIEmbeddings:** Cloud-based, high performance, requires API key and incurs cost.
- **HuggingFaceEmbeddings:** Can run locally, open-source, great for privacy and cost savings.
- **CohereEmbeddings, GooglePalmEmbeddings, etc:** Other provider-based models.

---

## LangChain Implementation Example

Here is a simple Python code snippet demonstrating how these concepts tie together in LangChain:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 1. Load your document (assuming you have a 'data.txt' file)
loader = TextLoader("data.txt")
document = loader.load()

# 2. Chunking 
# We use RecursiveCharacterTextSplitter which is character-based (not token-based) 
# because it preserves the semantic structure of the text better (paragraphs, sentences).
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Maximum size of each chunk
    chunk_overlap=200,    # Overlap between chunks to maintain context
    length_function=len   # Using standard python string length (characters)
)
chunks = text_splitter.split_documents(document)

print(f"Split document into {len(chunks)} chunks.")

# 3. Embedding
# Initialize the embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed the first chunk just to see the output
vector = embeddings_model.embed_query(chunks[0].page_content)
print(f"Dimension of the embedding vector: {len(vector)}") # e.g., 1536
```

### Summary of "Do we use tokens or not?"
While RAG pipelines ultimately feed tokens to the LLM, **we usually chunk by characters using `RecursiveCharacterTextSplitter` instead of token splitters.**

**Why?**
1. **Semantic Preservation:** Splitting by literal tokens might randomly cut a word in half or break a sentence at a grammatically unnatural point. Splitting by characters (specifically prioritizing `\n\n` then `\n`) respects the human-written structure (paragraphs and sentences), keeping the semantic meaning intact, which is crucial for good RAG retrieval.
2. **Speed & Simplicity:** Calculating string length in characters is computationally faster and simpler than running text through a tokenizer just to split it.

You only strictly use token-based splitters when you are pushing the absolute maximum limits of an LLM's context window and cannot afford to be off by even a few tokens.


## Example paragraphs

Para 1:  
Tokenization is a key step in natural language processing. It converts raw text into smaller units so models can process it efficiently.

Para 2:  
Different tokenization strategies exist depending on the use case. Some prioritize context preservation while others focus on strict size limits.

---

## How RecursiveCharacterTextSplitter works

1. **Start with largest unit (paragraphs)**
   - Split using `\n\n`
   - If chunk size ≤ limit → keep
   - If chunk size > limit → go deeper

2. **Fallback to sentences**
   - Split using `.`, `!`, `?`
   - If still too large → go deeper

3. **Fallback to words**
   - Split using spaces
   - If still too large → go deeper

4. **Final fallback (characters)**
   - Split at character level

---

## Key idea

It recursively splits text into smaller units **only when required**, preserving maximum context.

---

## Example behavior (small chunk size)

- Input: 2 paragraphs  
- If both fit → output = 2 chunks  
- If Para 1 too large → split into sentences  
- If sentence too large → split into words  

---

## Why it’s used

- Preserves semantic structure (paragraph → sentence → word)
- Maintains context
- Ideal for LLM pipelines (RAG, embeddings)

- Token ID:        same (if same token)
- Token embedding: different (depends on context)
- Chunk embedding: single vector representing all tokens
