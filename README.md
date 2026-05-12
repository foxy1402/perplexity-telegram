# NVIDIA + Exa Telegram Assistant

A grounded-answer Telegram bot built with:
- LangChain + NVIDIA NIM chat model with native tool calling
- Exa `/answer` endpoint (web-grounded answers) with multi-key rotation
- Single-tool agent: the LLM decides whether to answer directly or query the web
- Environment-variable-only configuration (container friendly)

## Model and Search Stack

- Chat model: `nvidia/nemotron-3-super-120b-a12b`
- Chat provider SDK: `langchain-nvidia-ai-endpoints`
- Web answers: Exa `/answer` via `exa-py`
- Exa keys: CSV from `EXA_API_KEYS` (example: `key1,key2,key3`)

## Architecture

Each user message goes through up to two LLM calls and (optionally) one Exa call:

1. **Orchestrator** — NVIDIA LLM sees the full chat history with a `web_answer(query)` tool bound. It either:
   - Answers the user directly from training data + context (no tool call), or
   - Emits a tool call with a fully self-contained query (pronouns resolved, current year stamped).
2. **Exa `/answer`** — fetches a fresh, web-grounded answer for the rewritten query. Multi-key rotation handles rate limits.
3. **Formatter** — NVIDIA LLM rewrites the grounded answer in Telegram-safe markdown. All citation artifacts (inline `[Site](url)` links, numeric `[1]` markers, `Sources:` footers, bare URLs) are stripped before and after the formatter pass — the user sees a clean, attribution-free answer.

The orchestrator handles multi-turn context: Exa is stateless, so the LLM is instructed to rewrite each follow-up question into a self-contained query that re-injects prior entities.

## Features

- Single-tool agent loop (no hand-rolled planner protocol or regex parsing)
- Multi-turn aware: pronouns/entities resolved by the orchestrator before search
- **Conversation memory** — older turns are summarized into a running memory note so long chats stay coherent past the context window
- Multi-key Exa rotation to maximize free-tier usage
- Retry and cancellation support (`/restart`)
- Telegram markdown-safe responses (no `##`, no LaTeX, no `|` tables, **no citations or source links** — clean answer only)
- Optional Telegram UID allowlist

## Requirements

- Python 3.10+
- Telegram bot token
- NVIDIA API key
- At least one Exa API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Required:

- `TELEGRAM_TOKEN` - Telegram bot token
- `NVIDIA_API_KEY` - NVIDIA NIM API key
- `EXA_API_KEYS` - Comma-separated Exa API keys

Optional:

- `TELEGRAM_ALLOWED_UIDS` - Preferred allowlist var. Comma-separated Telegram numeric user IDs allowed to chat
- `ALLOWED_USER_IDS` - Legacy fallback alias. Only used when `TELEGRAM_ALLOWED_UIDS` is empty
- `MAX_TOKENS` - Default: `4096`
- `REASONING_BUDGET` - Default: `16384`
- `TEMPERATURE` - Default: `0.7`
- `TOP_P` - Default: `0.95`
- `MAX_HISTORY_MESSAGES` - Default: `20` (hard cap; safety net if compaction fails)
- `EXA_TIMEOUT_SECONDS` - Default: `30` (per Exa /answer call)
- `EXA_ANSWER_MODEL` - Optional Exa model override (e.g. `exa`, `exa-pro`). Default: Exa picks
- `SESSION_TTL_SECONDS` - Default: `86400`
- `MAX_USER_SESSIONS` - Default: `5000`
- `LLM_MAX_CONCURRENCY` - Default: `8`
- `COMPACT_ENABLED` - Default: `true`. Set to `false` to disable conversation compaction
- `COMPACT_THRESHOLD_MESSAGES` - Default: `16`. Once conversation exceeds this many messages, the oldest turns get folded into a summary in the background
- `COMPACT_KEEP_RECENT_MESSAGES` - Default: `8`. After compaction this many recent messages stay verbatim; everything older is reduced to a summary
- `MEMORY_MAX_CHARS` - Default: `2000`. Hard cap on the running memory string length

## Example Environment

```env
TELEGRAM_TOKEN=123456:telegram-token
NVIDIA_API_KEY=nvapi-xxxxxxxx
EXA_API_KEYS=exa-key-a,exa-key-b,exa-key-c
TELEGRAM_ALLOWED_UIDS=111111111,222222222
MAX_TOKENS=4096
REASONING_BUDGET=16384
TEMPERATURE=0.7
TOP_P=0.95
EXA_TIMEOUT_SECONDS=30
```

## Run Locally

PowerShell:

```powershell
$env:TELEGRAM_TOKEN="123456:telegram-token"
$env:NVIDIA_API_KEY="nvapi-xxxxxxxx"
$env:EXA_API_KEYS="exa-key-a,exa-key-b,exa-key-c"
python bot.py
```

Bash:

```bash
export TELEGRAM_TOKEN="123456:telegram-token"
export NVIDIA_API_KEY="nvapi-xxxxxxxx"
export EXA_API_KEYS="exa-key-a,exa-key-b,exa-key-c"
python bot.py
```

## Commands

- `/start` - Show bot info and current settings
- `/status` - Show current model, thinking mode, web-access state, history length, memory size, and Exa key count
- `/help` - Show usage help
- `/web` - Show web-access state
- `/web on` - Enable Exa `/answer` tool
- `/web off` - Disable web access (model answers from training only)
- `/thinking on` - Include reasoning traces in output
- `/thinking off` - Hide reasoning traces
- `/clear` - Clear conversation history and memory
- `/restart` - Cancel a stuck/pending request

## How the Agent Works

1. User sends a message; bot appends it to the session history.
2. If web access is OFF: NVIDIA answers directly using only its training data + conversation history + memory.
3. If web access is ON:
   - NVIDIA is invoked with the `web_answer` tool bound. It sees the recent history verbatim plus the running memory (if any).
   - For greetings, self-capability questions, well-established facts, programming help, and follow-ups answerable from prior turns, NVIDIA returns a direct answer (no tool call).
   - For current events, prices, latest versions, news, or anything requiring fresh data, NVIDIA emits a `web_answer(query="...")` call. The query is rewritten to be self-contained.
   - The bot calls Exa `/answer`, then runs a second NVIDIA pass to format the grounded answer for Telegram.

## Conversation Memory (Compaction)

Long chats outgrow the LLM context window. Instead of silently dropping the oldest turns (the old behavior), the bot keeps a sliding-window summary:

- The most recent `COMPACT_KEEP_RECENT_MESSAGES` messages stay verbatim in history.
- Everything older is condensed into a `memory` string (max `MEMORY_MAX_CHARS`) by a summarizer LLM pass.
- That memory is appended to the orchestrator/formatter system prompts as "Conversation memory (earlier turns, summarized)" so the model still knows what was discussed.

Compaction runs **in the background after the bot delivers a response**, so the user never feels the extra latency. If the threshold has not been crossed (`COMPACT_THRESHOLD_MESSAGES`), the compaction call is a no-op. The mutation step has an atomic safety check that aborts cleanly if `/clear` or another message arrived during summarization.

Inspect the current memory size with `/status`. Use `/clear` to wipe both the history and the memory.

Disable the feature entirely with `COMPACT_ENABLED=false` — the bot reverts to plain hard-truncation behavior.

## UID Allowlist Notes

- Use `TELEGRAM_ALLOWED_UIDS` in new deployments.
- IDs must be Telegram numeric user IDs, not usernames.
- If `TELEGRAM_ALLOWED_UIDS` is not set or empty, the bot falls back to `ALLOWED_USER_IDS`.
- If both are empty, the bot is open to all users.

## Exa Key Rotation

- Keys are loaded from `EXA_API_KEYS` in memory.
- Requests use round-robin key selection.
- On transient/rate-limit failure, the bot tries the next key.
- No persistent disk required.

## Container Notes

This project is designed for platforms with ephemeral filesystems:

- No required local cache/database
- All configuration is from environment variables
- Safe to deploy as a single process container

Minimal start command:

```bash
python bot.py
```

## Security Notes

- Set `TELEGRAM_ALLOWED_UIDS` to restrict who can use the bot.
- Do not commit API keys.
- Rotate Exa and NVIDIA keys periodically.

## Troubleshooting

- Error: missing `TELEGRAM_TOKEN`
  - Set a valid Telegram bot token.
- Error: missing `NVIDIA_API_KEY`
  - Set your NVIDIA API key.
- Error: missing `EXA_API_KEYS`
  - Set one or more Exa keys in CSV format.
- Bot replies with empty output
  - Try `/restart`, then resend.
  - Reduce temperature or adjust token limits.
- Bot keeps calling the web for follow-ups that should be answerable from prior context
  - That is the LLM's call. If you find it over-searching, you can edit the `WHEN TO ANSWER DIRECTLY` section in `_PROMPT_ORCHESTRATOR` to bias it more.
