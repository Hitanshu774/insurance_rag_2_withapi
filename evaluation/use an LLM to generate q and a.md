<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>
```
# <q>Or use an LLM to generate questions and answers from each chunk, then lightly check a subset for correctness.</q> guide me through this
```

Here's a step-by-step guide to **auto-generate test questions and answers from your document chunks using an LLM**, then manually validate a subset:

***

## Step 1: Extract your document chunks

First, get all the text chunks from your RAG system. In your notebook:

```python
# Assuming you have a vector store or retriever already set up
# Get ALL documents (not just top-k for a query)

all_docs = vector_store.get_documents()  # or similar for your setup
# OR if you loaded them differently:
all_docs = list(vector_store._collection.get()['documents'])

print(f"Total chunks: {len(all_docs)}")
for i, doc in enumerate(all_docs[:3]):  # show first 3
    print(f"\nChunk {i}:")
    print(doc.page_content[:200])  # first 200 chars
```


***

## Step 2: Use an LLM to generate Q\&A pairs from chunks

Add this helper function to your notebook:

```python
from langchain_core.messages import SystemMessage, HumanMessage

def generate_qa_from_chunk(chunk_text: str, llm) -> dict:
    """
    Given a text chunk, ask an LLM to generate a Q&A pair.
    Returns: {"question": "...", "answer": "...", "keywords": [...]}
    """
    
    system_prompt = """You are an expert at creating test questions from documents.
Given a text chunk, generate:
1. A clear, factual question that can be answered from the chunk
2. The reference answer (2-3 sentences, based on the chunk)
3. 3-5 keywords that MUST appear in a correct answer

Output as JSON:
{
  "question": "Who founded X?",
  "reference_answer": "X was founded by Y in Z.",
  "keywords": ["Y", "founded", "X"]
}

Important:
- Make the question challenging but answerable from the text
- Include specific names, dates, numbers where present
- Keywords should be critical terms that prove understanding
"""

    user_message = f"Generate a Q&A pair from this chunk:\n\n{chunk_text}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    try:
        import json
        result = json.loads(response.content)
        return result
    except:
        print(f"Failed to parse LLM output: {response.content}")
        return None


def generate_test_set(all_docs, llm, num_chunks: int = None):
    """
    Generate Q&A pairs from document chunks.
    """
    if num_chunks:
        chunks = all_docs[:num_chunks]
    else:
        chunks = all_docs

    generated_tests = []
    
    for idx, doc in enumerate(chunks, 1):
        print(f"Generating Q&A for chunk {idx}/{len(chunks)}...")
        
        qa = generate_qa_from_chunk(doc.page_content, llm)
        
        if qa:
            qa["category"] = "auto_generated"  # you can categorize later
            qa["chunk_index"] = idx
            generated_tests.append(qa)
        
        # Be nice to the API—add a small delay if needed
        # time.sleep(0.5)
    
    return generated_tests
```


***

## Step 3: Generate Q\&A pairs (example)

In your notebook:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Use your LLM (Google, OpenAI, etc.)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Generate Q&A from first 10 chunks (for testing)
generated = generate_test_set(all_docs, llm, num_chunks=10)

print(f"\nGenerated {len(generated)} Q&A pairs")

# Show first 3
for i, qa in enumerate(generated[:3], 1):
    print(f"\nTest {i}:")
    print(f"  Q: {qa['question']}")
    print(f"  A: {qa['reference_answer']}")
    print(f"  Keywords: {qa['keywords']}")
```


***

## Step 4: Manually validate a subset

Now **manually check** a sample (e.g., first 5–10) to ensure quality:

```python
import json

# Write generated tests to a file for easy review
with open("generated_tests_raw.jsonl", "w") as f:
    for qa in generated:
        f.write(json.dumps(qa) + "\n")

print("Saved to generated_tests_raw.jsonl")

# Now manually review the first 5 in your notebook:
for i, qa in enumerate(generated[:5], 1):
    print(f"\n{'='*80}")
    print(f"TEST {i} - Review & Edit:")
    print(f"{'='*80}")
    print(f"Question: {qa['question']}")
    print(f"Reference Answer: {qa['reference_answer']}")
    print(f"Keywords: {qa['keywords']}")
    print("\n👉 Is this Q&A valid? Edit if needed, else mark OK")
    # Manually check: Does the Q match A? Are keywords in A?
```


***

## Step 5: Clean up and save validated set

After manual review, create a **validated** test set:

```python
# Manually fix any bad ones or remove them
validated_tests = []

for qa in generated:
    # Example: skip if keywords don't appear in answer
    answer_lower = qa['reference_answer'].lower()
    all_kw_present = all(kw.lower() in answer_lower for kw in qa['keywords'])
    
    if all_kw_present and len(qa['question']) > 10:  # basic sanity checks
        validated_tests.append(qa)
    else:
        print(f"Skipped: {qa['question']}")

print(f"\nValidated {len(validated_tests)} / {len(generated)} tests")

# Save as tests.jsonl
with open("tests.jsonl", "w") as f:
    for test in validated_tests:
        f.write(json.dumps(test) + "\n")

print("✅ Saved validated tests to tests.jsonl")
```


***

## Step 6: Use the generated test set with RAGEvaluator

Now you have a `tests.jsonl`, use it like before:

```python
from rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator("tests.jsonl")
report = evaluator.run_evaluation(get_rag_answer, verbose=False)
evaluator.print_report(report)
evaluator.save_results("evaluation_results.json")
```


***

## Example output

```
Generating Q&A for chunk 1/10...
Generating Q&A for chunk 2/10...
...

Generated 10 Q&A pairs

TEST 1 - Review & Edit:
================================================================================
Question: Who founded Insurellm?
Reference Answer: Avery Lancaster founded Insurellm in 2015 to revolutionize insurance.
Keywords: ['Avery', 'Lancaster', 'Insurellm']

👉 Is this Q&A valid? Edit if needed, else mark OK
...

Validated 9 / 10 tests
✅ Saved validated tests to tests.jsonl
```


***

## Tips

1. **Start small**: Generate from 5–10 chunks first, validate manually, then scale up.
2. **Lower temperature**: Use `temperature=0.5` or lower for more consistent, factual outputs.
3. **Better prompts**: Adjust the system prompt to match your domain (e.g., "Generate financial Q\&A pairs").
4. **Multiple LLMs**: Try different models (GPT-4, Gemini, Claude) to compare quality.
5. **Categories**: Add logic to auto-categorize (e.g., "who/what" → `direct_fact`, "when" → `temporal`).
6. **Deduplication**: Remove very similar questions to keep variety.

***

This approach lets you bootstrap a **test set from scratch** in under an hour, then iteratively improve your RAG based on real evaluation results.
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/73087562/4ee0c016-cd1c-414c-a244-04cf4b9646dd/one.ipynb

[^2]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/73087562/ad347c7f-cb68-4193-9bf5-b114bdfd7a1e/tests.jsonl

