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
3. **Formatter** — NVIDIA LLM rewrites the grounded answer in Telegram-safe markdown, stripping citation markers `[1]`, `[2]`, etc.

The orchestrator handles multi-turn context: Exa is stateless, so the LLM is instructed to rewrite each follow-up question into a self-contained query that re-injects prior entities.

## Features

- Single-tool agent loop (no hand-rolled planner protocol or regex parsing)
- Multi-turn aware: pronouns/entities resolved by the orchestrator before search
- Multi-key Exa rotation to maximize free-tier usage
- Retry and cancellation support (`/restart`)
- Telegram markdown-safe responses (no `##`, no LaTeX, no `|` tables, no `[N]` citations)
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
- `MAX_HISTORY_MESSAGES` - Default: `20`
- `EXA_TIMEOUT_SECONDS` - Default: `30` (per Exa /answer call)
- `EXA_ANSWER_MODEL` - Optional Exa model override (e.g. `exa`, `exa-pro`). Default: Exa picks
- `SESSION_TTL_SECONDS` - Default: `86400`
- `MAX_USER_SESSIONS` - Default: `5000`
- `LLM_MAX_CONCURRENCY` - Default: `8`

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
- `/status` - Show current model, thinking mode, web-access state, history length, and Exa key count
- `/help` - Show usage help
- `/web` - Show web-access state
- `/web on` - Enable Exa `/answer` tool
- `/web off` - Disable web access (model answers from training only)
- `/thinking on` - Include reasoning traces in output
- `/thinking off` - Hide reasoning traces
- `/clear` - Clear conversation history
- `/restart` - Cancel a stuck/pending request

## How the Agent Works

1. User sends a message; bot appends it to the session history.
2. If web access is OFF: NVIDIA answers directly using only its training data + conversation history.
3. If web access is ON:
   - NVIDIA is invoked with the `web_answer` tool bound.
   - For greetings, self-capability questions, well-established facts, programming help, and follow-ups answerable from prior turns, NVIDIA returns a direct answer (no tool call).
   - For current events, prices, latest versions, news, or anything requiring fresh data, NVIDIA emits a `web_answer(query="...")` call. The query is rewritten to be self-contained.
   - The bot calls Exa `/answer`, then runs a second NVIDIA pass to format the grounded answer for Telegram.

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
