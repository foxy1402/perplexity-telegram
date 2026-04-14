# NVIDIA + Exa Telegram Assistant

A Perplexity-style Telegram bot built with:
- LangChain + NVIDIA NIM chat model
- Exa search retrieval with multi-key rotation
- Iterative search loop (multiple SEARCH rounds before FINAL answer)
- Environment-variable-only configuration (container friendly)

## Model and Search Stack

- Chat model: `nvidia/nemotron-3-super-120b-a12b`
- Chat provider SDK: `langchain-nvidia-ai-endpoints`
- Search provider: Exa via `langchain-exa`
- Exa keys: CSV from `EXA_API_KEYS` (example: `key1,key2,key3`)

## Features

- Single-provider architecture (NVIDIA only)
- Accuracy-first loop: model can search multiple times before answering
- Web search enabled by default
- Exa API key rotation to maximize free-tier usage
- Retry and cancellation support (`/restart`)
- Telegram markdown-safe responses
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
- `EXA_MAX_RESULTS` - Default: `5`
- `EXA_MAX_SNIPPET_LEN` - Default: `500`
- `EXA_TIMEOUT_SECONDS` - Default: `20`
- `RESEARCH_MAX_STEPS` - Default: `4`
- `RESEARCH_MAX_SNIPPETS` - Default: `20`

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
EXA_MAX_RESULTS=5
EXA_MAX_SNIPPET_LEN=500
EXA_TIMEOUT_SECONDS=20
RESEARCH_MAX_STEPS=4
RESEARCH_MAX_SNIPPETS=20
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

- `/start` - Show bot status and quick help
- `/help` - Show usage help
- `/web` - Show web-search state
- `/web on` - Enable Exa search
- `/web off` - Disable web search
- `/thinking on` - Include reasoning traces in output
- `/thinking off` - Hide reasoning traces
- `/clear` - Clear conversation history
- `/restart` - Cancel a stuck/pending request

## How Search Works

1. For trivial greetings/short messages, bot answers directly.
2. For normal messages, bot enters an iterative planner loop.
3. At each step, the model decides one action:
   - `SEARCH: query` (continue research)
   - `FINAL: answer` (stop and answer)
4. The loop can repeat up to `RESEARCH_MAX_STEPS`.
5. Exa results are accumulated and used for final synthesis when needed.

## UID Allowlist Notes

- Use `TELEGRAM_ALLOWED_UIDS` in new deployments.
- IDs must be Telegram numeric user IDs, not usernames.
- If `TELEGRAM_ALLOWED_UIDS` is not set or empty, the bot falls back to `ALLOWED_USER_IDS`.
- If both are empty, the bot is open to all users.

## Exa Key Rotation

- Keys are loaded from `EXA_API_KEYS` in memory.
- Requests use round-robin key selection.
- On transient/rate-limit failure, bot tries the next key.
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
