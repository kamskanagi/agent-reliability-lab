# Experiment README template

Every experiment folder (`01_prompt_regression/`, `02_rag_retrieval_eval/`, …) uses
exactly this README structure. Copy the block below into the experiment's `README.md`
and replace the example content (a tool-calling failure test).

~~~markdown
# Tool-Calling Failure Test

Test how an AI agent behaves when its tool returns malformed, delayed or unavailable responses.

## Why this matters

Agents don't fail only because the model is wrong.
Production systems also fail when tools are unavailable.

## Architecture

[diagram]

## Experiment

**Baseline:** model + tool

**Failure injected:**
- HTTP 500
- Timeout
- Malformed JSON
- Missing fields

## Run it

```bash
git clone ...
pip install -r requirements.txt
python run.py
```

## Results

| Scenario | Success | Latency | Retries |
|----------|---------|---------|---------|

## What surprised me

...

## Production takeaway

...

## Try another experiment

- → RAG retrieval failure
- → Prompt regression
~~~
