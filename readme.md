# Ola Domain Support Agent: Capstone

**Track completed: Ola (Business Operations / Customer Support)**

A support agent that answers Ola policy questions from a knowledge base, looks up support-ticket status, remembers a conversation, is guarded against PII leaks and prompt injection, and is served over FastAPI. Everything runs under `MOCK_LLM` with zero API keys, zero paid accounts, and no network calls at inference time. Embeddings (SentenceTransformers `all-MiniLM-L6-v2`) and the vector index (ChromaDB) run locally.

## Setup and run

```bash
python -m venv venv
venv\Scripts\activate                 # Windows
pip install -r requirements.txt
copy .env.example .env                # MOCK_LLM=mock is already set

python setup_kb.py                    # writes the 12 knowledge-base files
python dataset.py                     # Task 1: dataset + validation report
python rag/chunker.py                 # Task 3: chunk counts
python rag/indexer.py                 # Task 3: builds both ChromaDB collections
python rag/retriever.py               # Task 4: threshold calibration + grounded demo
python rag/evaluator.py               # Task 5: precision/recall, both collections
python agents/tools.py                # Tasks 3-6: tool demos
python agents/guardrails.py           # Task 10: PII / injection / groundedness demos
python agents/memory.py               # Task 8: multi-turn + fresh session
python agents/crew.py                 # Tasks 7-10: crew run, 4 demos
python eval/evaluator.py              # Task 13: 15-query judge evaluation
python review/demo.py                 # Task 14: AutoGen review, approve + revise
python governance/demo_least_autonomy.py   # Task 15: least autonomy
python governance/demo_budget.py      # Task 15: budget cap
python governance/demo_cache.py       # Task 16: caching
uvicorn api.main:app --port 8000      # Tasks 11-12: HTTP + WebSocket API
python ws_test.py                     # Task 11: WebSocket + disconnect test
python chat.py                        # optional: interactive terminal chat
```

**Telemetry is disabled.** `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` are set in `.env.example` and again in code before any `crewai` import (`agents/crew.py`, `api/main.py`, `chat.py`).

## Repository layout

```
dataset.py            Task 1 generator + escalation_score (Task 6)
setup_kb.py           writes the 12 knowledge-base documents
knowledge_base/       12 policy documents (Task 2)
rag/                  chunker, indexer, retriever, evaluator (Tasks 3-5)
agents/               tools, mock_llm, crew, memory, schemas, guardrails (Tasks 6-10)
api/                  FastAPI app + JSON-Lines logging (Tasks 11-12)
eval/                 judge.py + evaluator.py (Task 13)
transcripts/          graded evidence for every task
logs/requests.jsonl   structured request log
review/
governance/
chat.py
ws_test.py
```

## Task 1: Dataset design choices (reproducible)

| Parameter | Value |
|---|---|
| Seed | 42 |
| Records | 50 |
| Category weights | Billing 0.25, Technical Issue 0.25, Account Access 0.20, Product Defect 0.15, General Inquiry 0.15 |
| Status weights | Open 0.15, In Progress 0.25, Escalated 0.10, Resolved 0.30, Closed 0.20 |
| `resolution_time_hours` range, per category | Billing 2-72, Technical Issue 1-48, Account Access 0.5-24, Product Defect 4-96, General Inquiry 0.5-12 |
| `days_since_created` | uniform integer 0-30 |
| Escalation probability | 0.12 base; 0.80 if status = Escalated; 0.25 if days_since_created > 20 |

Reasoning for the ranges: billing disputes and defect QA loops take days, while policy look-ups resolve in hours.

Verified output (seed 42):

- Billing 11, Technical Issue 15, Account Access 6, Product Defect 11, General Inquiry 7 (every category has at least 3).
- Open 7, In Progress 15, Escalated 2, Resolved 15, Closed 11 (every status present).
- `escalated=True` for 11 of 50 records = **22.0%**, inside the 10-30% band. No record was hand-edited.

Note: only 2 tickets have status `Escalated`, while 11 have the boolean `escalated=True`. The generator draws status and the flag separately, so a ticket can be flagged for escalation without currently sitting in the `Escalated` status.

## Task 2: Knowledge base

12 documents in `knowledge_base/`, two paragraphs each, covering: ticket-priority classification, SLA by severity, escalation matrix, refund/compensation, communication channels, business hours/holidays, repeat complaints, service credits, feedback collection, VIP handling, outage communication, and data retention.

## Task 3: Chunking and indexing

| Strategy | Setting | Collection |
|---|---|---|
| Fixed-size with overlap | 300 characters, 50 overlap | `ola_fixed` |
| Sentence-based | 3 sentences per chunk | `ola_sentence` |

Each strategy is embedded with `all-MiniLM-L6-v2` and indexed into its own ChromaDB collection with `collection.upsert()`.

## Task 4: "I don't know" threshold (measured, not a preset)

Top-1 cosine similarity measured against the fixed-size collection:

| Type | Query | Top-1 similarity |
|---|---|---|
| In-scope | What is the SLA for Priority 1 tickets? | 0.6811 |
| In-scope | How do I request a refund from Ola? | 0.6952 |
| In-scope | When should a ticket be escalated to Tier 2? | 0.5832 |
| Out-of-scope | What is the weather in Mumbai today? | 0.1961 |
| Out-of-scope | Tell me a joke about taxi drivers | 0.2804 |

Across all 12 in-scope evaluation queries, top-1 similarity ranged from 0.5446 to 0.8296. The out-of-scope cluster tops out at 0.2804, so the gap between the clusters is 0.28 to 0.54. **Chosen threshold: 0.45**, inside that gap. The out-of-scope weather query triggers the fallback.

## Task 5: Chunking comparison (document level)

Chunks are mapped back to parent documents and de-duplicated before scoring. `rag/evaluator.py` prints, per query, the relevant documents, the retrieved documents, TP, precision, and recall, for both collections.

| Strategy | Avg precision | Avg recall |
|---|---|---|
| Fixed-size | 0.5000 | 1.0000 |
| Sentence | 0.4167 | 1.0000 |

Example arithmetic, fixed-size, "What is the SLA for Priority 1 tickets?": retrieved {01_ticket_priority_classification, 02_sla_by_severity}, relevant {02_sla_by_severity}, TP = 1, precision = 1/2 = 0.50, recall = 1/1 = 1.00.

**Recommendation: fixed-size.** It has higher average precision (0.5000 vs 0.4167) at identical recall (1.0000), so it pulls in fewer unrelated documents next to the correct one. The out-of-scope weather query is included in these averages with an empty relevant set (precision 0, recall set to 1.0 by convention).

## Task 6: Escalation score

```
escalation_score = 0.6 * escalated + 0.4 * min(days_since_created / 26, 1.0)
recommend escalation if escalation_score > 0.5
```

Justification from the generated data:

- The 80th percentile of `days_since_created` is **26.2 days**, so recency is capped at 26 days. About 80% of tickets fall below the cap, so the recency term separates tickets instead of saturating.
- An earlier 14-day cap was rejected. Only 36% of tickets are 14 days old or younger, so that cap gave 64% of tickets the same recency value.
- A non-escalated ticket scores at most 0.4, so age alone never crosses the 0.5 threshold.
- Every escalated ticket scores at least 0.6, and the recency term ranks urgency within that group. Example: OLA-0006 (escalated, 3 days old) scores 0.6 + 0.4 x 3/26 = 0.646.
- In this dataset 11 of 50 tickets are escalated.

## Task 7: How the tools are invoked (design decision, stated openly)

With CrewAI 1.15 and a custom `BaseLLM`, a ReAct-formatted string from the mock LLM was not dispatched to tools. The agent log shows `todos_count=0`, and the `Action:` text came back as the final answer. So `run_crew()` invokes `rag_policy_lookup` and `check_support_ticket_status` explicitly, logs each call with a `[Tool Invocation]` line, and injects the tool output into the Retrieval and Lookup agents' task descriptions before `crew.kickoff()`.

- Crew: Retrieval Agent (RAG tool only), Lookup Agent (ticket tool only), Response Composer (no tools), run in sequential order.
- Under `MOCK_LLM` the Composer returns a fixed string. The evidence of retrieval is the tool output visible in the task logs, not the final sentence.
- `MockLLM` extends `crewai.llms.base_llm.BaseLLM`. It parses only the model's own generated messages for observations (not the system-prompt template) and dispatches tools by declared argument schema, not by tool name.

## Task 8: Session memory

Per-session history is held in LangChain `ChatMessageHistory`, in process memory only. The `LangChainDeprecationWarning` is expected and not silenced. `python agents/memory.py` prints a three-turn session where history accumulates, then a fresh session that starts empty and stays separate from the first.

## Task 9: Structured output

Every crew response is validated against the Pydantic model `CrewResponse` (`agents/schemas.py`) with `validate_response()` before it is returned.

## Task 10: Guardrails

| Guardrail | Behaviour | Demonstrated |
|---|---|---|
| Phone-number masking (input) | Replaces phone numbers with `[PHONE_REDACTED]` before the crew sees them | Crew demo 3, API log check |
| Prompt-injection detection (input) | Blocks the request before the crew starts | Crew demo 4 |
| Groundedness check (output) | Refuses when top-1 similarity is below 0.45 | `rag/retriever.py` weather query, `agents/guardrails.py` demo |

Name, address, and payment details are free text with no fixed format and are out of scope for the keyless masker, as the brief states.

## Task 11: FastAPI

- `POST /ask` and `POST /add-document` with Pydantic request and response models, plus `GET /health`.
- `WebSocket /ws/chat/{session_id}` for multi-turn chat. `WebSocketDisconnect` is caught, the server keeps running, and a second client still gets answers.
- The blocking crew call runs in a worker thread (`def` endpoint for `/ask`, `run_in_threadpool` for the WebSocket), because CrewAI 1.15 refuses to run `kickoff()` inside a running event loop.
- Evidence: `ws_test.py` output and `transcripts/task11_server_output.txt`, which shows `[WebSocket] Client disconnected: session=ws_test`.

## Task 12: Structured logging

Every request writes one JSON-Lines entry to `logs/requests.jsonl` with `timestamp`, `trace_id`, `session_id`, masked `query`, `pii_redacted`, `duration_ms`, `grounded`, `agent_path`, `blocked`, and `has_ticket`. The logger masks the query itself before writing. Check: after a request containing `+91-9876543210`, the log shows `[PHONE_REDACTED]`, and searching the file for `9876543210` finds nothing.

## Task 13: Evaluation (LLM-as-judge, under MOCK_LLM)

For each of 15 queries, `eval/judge.py` builds a judge prompt (query, expected and retrieved source, similarity, retrieved context, system response). A deterministic `MockJudge` reads that prompt and returns JSON scores, standing in for a real LLM judge. The prompt and the evaluator would not change if a real model replaced the mock.

- 12 queries cover the 12 required knowledge-base topics, plus 3 edge cases: 1 out-of-scope weather query, 1 injection attack, 1 out-of-scope joke.
- Grounding requires similarity above the threshold **and** a correct source.

| Metric | Average over 15 queries |
|---|---|
| Accuracy | 0.9333 |
| Grounding | 0.9333 |
| Completeness | 0.9333 |
| Safety | 1.0000 |

**Known failure:** "How does Ola handle repeat complaints?" retrieves `11_outage_communication_protocol` (similarity 0.5446) instead of `07_repeat_complaint_handling`, scoring 0 on Accuracy, Grounding, and Completeness. A likely cause is vocabulary mismatch (the query says "complaints", the document says "complainant"), combined with 300-character chunks splitting that document's key sentences. This was not verified by experiment. The query was left unchanged instead of being tuned to pass.

Because the judge is a deterministic rule set, it measures retrieval quality, not answer fluency.

## Part 4: Resilience and governance

### Task 14: AutoGen review stage

- `review/autogen_review.py` builds a two-agent `RoundRobinGroupChat`: `Policy_Compliance_Reviewer` and `Final_Editor`, bounded with `max_turns=2` so each agent speaks once.
- `Final_Editor` uses `output_content_type=Verdict` (fields `approved: bool`, `final_answer: str`, `reason: str`), and the team is built with `custom_message_types=[StructuredMessage[Verdict]]`, which is required for structured messages to be registered.
- Under `MOCK_LLM` the agents use `MockChatCompletionClient` (`review/mock_model_client.py`), a rule-based implementation of AutoGen's `ChatCompletionClient`. The reviewer flags a draft sentence as unsupported when fewer than 60% of its content words appear in the retrieved context. The editor turns those flags into the verdict.
- Demonstrated on 2 queries (`python review/demo.py`), with the context taken from the real retriever:
  - "What is the SLA for Priority 1 tickets?": clean draft, `approved=True`, draft returned unchanged.
  - "How do I request a refund from Ola?": a draft with an injected claim ("free airport transfer voucher"), `approved=False`, the injected sentence removed.
- The demo drafts are hand-written test fixtures, because under `MOCK_LLM` the Composer always returns one fixed sentence with no claims to review.
- Limits: the check is lexical word overlap, not semantic understanding, and the review stage is demonstrated on its own rather than inside the `/ask` pipeline.

### Task 15: Four-layer governance (two layers enforced and demonstrated)

**Application layer: least autonomy.** `check_support_ticket_status` is restricted to the Lookup Agent. `governance/least_autonomy.py` holds a tool-to-role allow-list. `attach_tools()` refuses to give a restricted tool to any other agent and raises `PermissionError`, and `enforce_least_autonomy()` audits all agents when `agents/crew.py` is imported, so a wiring mistake stops the crew from starting. Demonstrated by attempting to attach the tool to the Composer: the attempt is blocked and the Composer's tool list stays empty (`transcripts/task15_least_autonomy.txt`).

**Application layer: risk level, Medium.** The brief's scheme lists customer-support tickets under Medium, and this is a customer-support agent. It answers policy questions and reports ticket status and an escalation recommendation. It handles customer messages that may include a phone number, which the input guardrail masks before the crew or the log sees it. It is not High because it handles no medical data, makes no hiring decisions, and holds no financial account data. It also takes no autonomous action: it issues no refunds, changes no accounts, and only recommends escalation. It would move to High if it started executing refunds or handling payment details.

**Runtime layer: per-request budget.** `governance/budget_cap.py` estimates tokens as characters / 4 plus a 600-token reserve for tool output and prompts, and rejects any request over 1,500 tokens ($0.003 at an illustrative $0.002 per 1K tokens) before any tool call or crew run. Demonstrated: a normal query estimates at 609 tokens and is accepted; a 10,434-character query estimates at 3,208 tokens ($0.006416) and is rejected with `agent_path: rejected` and no tool or crew activity (`transcripts/task15_budget_cap.txt`). Conversation history is included in the estimate, so long sessions are also bounded. The token count is an estimate and the price is illustrative, because `MOCK_LLM` costs nothing.

### Task 16: Response caching

`governance/cache.py` is an in-memory cache around the grounded-generation step, keyed by normalized query text (lowercased, whitespace collapsed, trailing punctuation removed). Because the tool only receives the PII-masked query, keys never contain a raw phone number. The cache is cleared whenever `/add-document` changes the knowledge base, so it cannot serve stale answers.

Before/after evidence (`transcripts/task16_caching.txt`): three requests (cold, identical repeat, messy variant) caused **1** real retriever call. Cold call 169.18 ms; cached calls 0.04 ms and 0.02 ms. Cache stats: 2 hits, 1 miss, 1 entry.

## Known limitations

- Under `MOCK_LLM` the Composer always returns the same sentence, so the API's `answer` field does not show retrieved content. The retrieved policy text is visible in the crew's task logs.
- `escalation_score` is not returned in the `/ask` response body. It appears in the crew's lookup output.
- Pure-retrieval scores come from a small 12-document knowledge base and say nothing about performance on a larger one.
- The AutoGen review stage runs as a standalone demo and is not part of the `/ask` pipeline.

## Transcripts

## Transcripts

All text evidence is in `transcripts/`:

| Task | File |
|---|---|
| 1 Dataset | `task01_dataset.txt` |
| 2 Knowledge base | the `knowledge_base/` folder |
| 3 Chunking + indexing | `task03_chunking.txt`, `task03_indexing.txt` |
| 4 Grounded generation + threshold | `task04_grounded_generation.txt` |
| 5 Chunking comparison | `task05_chunking_eval.txt` |
| 6 Lookup tool + escalation score | `task06_tools.txt` |
| 7 Crew (both tools invoked) | `task7_crew_demo.txt` |
| 8 Memory (multi-turn, then fresh) | `task8_memory_multiturn.txt`, `task8_memory_fresh.txt` |
| 9 Structured output | `task09_schema.txt` (schema demo); every crew run in `task7_crew_demo.txt` is validated against it |
| 10 Guardrails | `task10_guardrails.txt`, plus demos 3 and 4 in `task7_crew_demo.txt` |
| 11 FastAPI + WebSocket | `task11_http_endpoints.txt`, `task11_websocket.txt`, `task11_server_output.txt` |
| 12 Structured logging | `task12_logging.txt`, `logs/requests.jsonl` |
| 13 Evaluation | `task13_eval.txt` |
| 14 AutoGen review | `task14_autogen_review.txt` |
| 15 Governance | `task15_least_autonomy.txt`, `task15_budget_cap.txt` |
| 16 Caching | `task16_caching.txt` |