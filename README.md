Personal project by Somya Goyal, built while applying to Michelin. Not affiliated with or endorsed by Michelin; uses public/synthetic data.

# Tyre care & fitment assistant, with answer evaluation

A small support chatbot backend that answers tyre-care questions from a document set (RAG), decodes and compares tyre sizes with a tool, and refuses or escalates unsafe and off-topic questions. It also includes an evaluation harness that measures how accurate and reliable its answers are, on a dev set and on a held-out set written after the code was frozen.

## Why this matters for Michelin

- **Problem:** tyre-care and fitment questions ("what does 91V mean?", "can I fit this size?", "is this bulge OK?") come up again and again, and a wrong answer can be a safety issue.
- **Who feels it:** dealer and customer-support teams who answer these questions, and drivers who need a quick, safe answer. Michelin is a tyre manufacturer with operations in India, including Pune.
- **What this shows:** the skills the Junior GenAI Data Scientist role lists: an agent workflow, RAG, a structured-data tool, logging and error handling, and **testing chatbot answers for accuracy and reliability**. The assistant never tells anyone a damaged tyre is safe to drive.

No impact numbers are claimed. All numbers below were measured on my own run of this repo.

## Demo (offline mode, real output)

```text
$ python -m assistant "can I fit 225/45 R17 94W on a car that needs 91V?"
[route: tool]
Yes, it is a suitable match: 225/45 R17 94W vs required 91V. Reasons: load index 94 (670 kg) meets or
exceeds the required 91 (615 kg); speed rating W (270 km/h) meets or exceeds the required V (240 km/h).
Always confirm the size is approved in your vehicle handbook.

Sources: size_marking, load_index, speed_rating

$ python -m assistant "My tyre has a bulge on the sidewall, is it safe to drive?"
[route: safety]
This sounds safety-critical. Please do not drive on this tyre. Fit the spare or call roadside assistance,
and have a qualified tyre professional inspect it before any further use. I can't confirm a damaged tyre
is safe remotely.

Sources: damage_bulge, mixing_tyres

$ python -m assistant "How often should I rotate my tyres?"
[route: rag]
Rotating tyres every 8,000 to 10,000 km helps them wear evenly and last longer. Directional tyres may only
be swapped front-to-back on the same side. Check your handbook, because some vehicles with different front
and rear sizes cannot be rotated.

Sources: rotation
```

Each question also writes a structured JSON log line to stderr: `{"event": "ask", "q": ..., "route": "rag", "docs": ["rotation"], "latency_ms": 2.1}`.

## Approach

```
question -> classify (rules) -> tool          : parse/decode tyre size, check fitment, synthetic catalogue lookup
                             -> rag           : TF-IDF top-3 -> answer generator -> cite doc ids
                             -> safety        : fixed "do not drive, see a professional" + cite docs
                             -> out_of_scope  : polite refusal (pricing, medical, legal, off-topic,
                                                or retrieval score below threshold)
```

- **Agent graph:** built with **LangGraph** (`StateGraph` with a conditional edge from `classify` to four nodes). If LangGraph is not installed, `assistant/agent.py` falls back to a hand-written router with the same behaviour. The classifier itself is rule-based (regex), so it is predictable and testable. Safety is checked first.
- **Knowledge base:** `data/kb/` has 13 short markdown docs. They are a **synthetic knowledge base written for this demo; not official manufacturer guidance**. They contain generic, widely known tyre-safety advice in my own words.
- **Retrieval:** scikit-learn TF-IDF over stemmed unigrams and bigrams, with a light suffix-stripping stemmer and no extra dependencies.
- **Answer generation:** the default `ScriptedLLM` is extractive. It picks the most relevant sentences from the top two docs, so it runs offline and gives the same answer every time. `OpenAICompatibleLLM` is the real-LLM path, behind the same `generate(question, docs)` interface. Set `ASSISTANT_LLM=openai` and `OPENAI_API_KEY` to use it (`OPENAI_BASE_URL` works for any OpenAI-compatible endpoint, such as Gemini). Tests never use the network.
- **Tool:** `assistant/tyre_size.py` parses sizes such as `205/55 R16 91V` or `225/45ZR17`. It decodes load index to kg using a subset of the standard table, and speed rating to km/h. It checks fitment on load index, speed rating, rim diameter, and overall diameter (within 3%). It also queries `data/tyre_specs.csv`, a synthetic catalogue with SKUs `SYN-001`...
- **Reliability:** empty input is handled, exceptions are caught and logged with a safe message, the API validates input, and a missing real LLM falls back to offline mode.

## Evaluation

`python run_eval.py` runs both question sets, prints the table, and writes `results/eval_report.md`. It also lists every failure.

| metric | what it measures |
|---|---|
| route_accuracy | the router chose the expected route (tool / rag / safety / out_of_scope) |
| retrieval_hit@3 | for KB questions, an expected doc is in the top-3 retrieved |
| key_fact_accuracy | every required fact string (for example "615 kg", "1.6 mm", "do not drive") appears in the answer |
| unsafe_refusal_rate | share of safety or out-of-scope questions that were refused or escalated |
| false_refusal_rate | share of answerable questions that were wrongly refused |
| tool_correctness | the tool action and verdict (for example `fits: false`) match the expected result |

**Process (in this order):** (1) I built the system and iterated only against `eval/dev_questions.jsonl` (26 questions). (2) I froze the code: the sha256 of `assistant/*.py` + KB + CSV was `024addd8...`. (3) I then wrote `eval/heldout_handwritten.jsonl` (18 questions with different wording, typos, Hinglish, tricky fitment and unsafe cases). (4) I ran the evaluation **once**. I made no code changes after seeing the held-out results.

### Results (measured, offline mode)

| metric | dev (n=26) | held-out (n=18) |
|---|---|---|
| route_accuracy | 1.000 | 0.889 |
| retrieval_hit@3 | 1.000 | 0.875 |
| key_fact_accuracy | 0.923 | 0.778 |
| unsafe_refusal_rate | 1.000 | 0.833 |
| false_refusal_rate | 0.000 | 0.000 |
| tool_correctness | 1.000 | 0.333 (1/3) |
| mean latency | ~2.3 ms | ~2.1 ms |

**v1.1 (one post-freeze fix):** after the blind run I fixed the h10 bug below (size roles are now read from cue words like "manual says", "needs", "instead of", "offering") and added a regression test. Re-running gives held-out tool correctness 0.667 (2/3) and key-fact accuracy 0.833; the other metrics are unchanged. The h10 case is no longer blind, so the v1.0 column above stays the honest blind result.

The held-out set scores lower than the dev set, as expected. These are the held-out failures:

- **h10, a real bug with a safety impact:** for "Manual says 195/65 R15 **91H**. Shop is offering ... **91T**, is that fine?" the assistant answered **yes**. The tool assumes the first size mentioned is the candidate. Here the user gave the required size first, so the comparison ran backwards. **Fixed in v1.1** by reading cue words ("manual says / needs / instead of / offering"); see `_assign_roles` in `assistant/agent.py` and `test_required_size_first_is_not_reversed`.
- **h11, my labelling error, not a system error:** "215/60R16 95V instead of 205/55R16 91V". I labelled it as fitting, but the overall diameter differs by +5.1%, so the tool's "not suitable" answer is correct. I kept the label as run so the numbers are not edited after the run. With the label corrected, tool correctness would be 2/3 and key-fact accuracy 0.833. This shows that eval sets need review too.
- **h18, an out-of-scope leak:** "write me a python script to sort a list" was routed to RAG. The word "list" matched "lists" in `pressure_effects`, which gave a score of 0.10, just above the 0.08 refusal threshold. A single lexical threshold is a weak out-of-scope detector.
- **h12:** "Is a 94 load index OK if my car needs 98?" went to RAG instead of the tool, because bare load indices without a speed letter are not parsed. The RAG answer still contained the right principle (a lower load index is not acceptable).
- **h03:** for "max PSI written on the side", the top hit was `mixing_tyres` (it matched "side"). The answer missed the key point that the sidewall number is a limit, not a target.
- **h08:** "what does the letter after 91 mean" retrieved `size_marking` rather than `speed_rating`. The answer was usable, but the expected doc was not in the top-3.
- Dev misses: **d04** (the extractive answer picked the over-inflation sentence for an under-inflation question) and **d14** (the key phrase "maximum load" was not among the selected sentences).

## How to run

```bash
pip install -r requirements.txt
pytest                                   # 11 tests, offline
python -m assistant "what does 205/55 R16 91V mean?"
python run_eval.py                       # dev + held-out table, writes results/eval_report.md
uvicorn app:app --reload                 # POST /ask {"question": "..."}; GET /health
```

## Files

- `assistant/router.py`: rule-based intent classifier (safety, out_of_scope, tool, rag)
- `assistant/agent.py`: LangGraph graph, nodes, logging, error handling
- `assistant/tyre_size.py`: size parser, load/speed tables, fitment check, catalogue lookup
- `assistant/retriever.py`: KB loader and TF-IDF retriever
- `assistant/llm.py`: `ScriptedLLM` (offline) and `OpenAICompatibleLLM` behind one interface
- `run_eval.py`, `eval/*.jsonl`, `results/eval_report.md`: evaluation harness, question sets and latest report
- `app.py`: FastAPI app; `tests/`: pytest suite

## Limitations and next steps

- **Real LLM:** add a real LLM for answer writing, and add faithfulness checks (does every sentence come from a cited doc?), for example with RAGAS or an LLM-as-judge.
- **Router:** replace the regex router with a small trained classifier or LLM tool-calling. Keep the rule-based safety check as a hard guard that runs first.
- **Fitment tool:** the cue-word role detection (v1.1) is rule-based; ask the user to confirm when no cue is present. Also support bare load indices (h12).
- **Larger KB:** use chunking and embedding retrieval (FAISS/Chroma) with hybrid BM25, so paraphrases like "max PSI" match "maximum pressure".
- **Multilingual:** Hindi/Hinglish support today is accidental (it works only when English keywords appear). Add transliteration or a multilingual embedding model, and grow the held-out set with real dealer and support phrasing.
- **Data:** the KB and catalogue are synthetic. A real deployment would need reviewed, official content and human escalation.
