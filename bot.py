import asyncio
import datetime
import json
import logging
import os
import re
import threading
import time
import warnings
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Suppress LangChain advisory about <think> tag handling — our code already reads
# reasoning content from additional_kwargs['reasoning_content'] directly.
warnings.filterwarnings(
    "ignore",
    message="Reasoning content was parsed from <think> tags",
    category=UserWarning,
)

from exa_py import Exa
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Core environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EXA_API_KEYS = [
    k.strip() for k in os.getenv("EXA_API_KEYS", "").split(",") if k.strip()
]


def _parse_uid_allowlist() -> List[str]:
    """Parse Telegram UID allowlist from env.

    Supported env vars:
    - TELEGRAM_ALLOWED_UIDS (preferred)
    - ALLOWED_USER_IDS (backward compatible)
    """
    raw = os.getenv("TELEGRAM_ALLOWED_UIDS", "").strip()
    if not raw:
        raw = os.getenv("ALLOWED_USER_IDS", "").strip()
    if not raw:
        return []

    allowlist = []
    for item in raw.split(","):
        uid = item.strip()
        if not uid:
            continue
        if not uid.isdigit():
            logger.warning("Ignoring invalid Telegram UID in allowlist: %s", uid)
            continue
        allowlist.append(uid)
    return allowlist


ALLOWED_USER_IDS = _parse_uid_allowlist()

try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
except ValueError:
    MAX_TOKENS = 4096

try:
    REASONING_BUDGET = int(os.getenv("REASONING_BUDGET", "16384"))
except ValueError:
    REASONING_BUDGET = 16384

try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
except ValueError:
    TEMPERATURE = 0.7

try:
    TOP_P = float(os.getenv("TOP_P", "0.95"))
except ValueError:
    TOP_P = 0.95

try:
    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
except ValueError:
    MAX_HISTORY_MESSAGES = 20

try:
    EXA_TIMEOUT_SECONDS = float(os.getenv("EXA_TIMEOUT_SECONDS", "30"))
except ValueError:
    EXA_TIMEOUT_SECONDS = 30.0

EXA_ANSWER_MODEL = os.getenv("EXA_ANSWER_MODEL", "").strip() or None

try:
    SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
except ValueError:
    SESSION_TTL_SECONDS = 86400

try:
    MAX_USER_SESSIONS = int(os.getenv("MAX_USER_SESSIONS", "5000"))
except ValueError:
    MAX_USER_SESSIONS = 5000

try:
    LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
except ValueError:
    LLM_MAX_CONCURRENCY = 8

COMPACT_ENABLED = os.getenv("COMPACT_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

try:
    COMPACT_THRESHOLD_MESSAGES = int(os.getenv("COMPACT_THRESHOLD_MESSAGES", "16"))
except ValueError:
    COMPACT_THRESHOLD_MESSAGES = 16

try:
    COMPACT_KEEP_RECENT_MESSAGES = int(os.getenv("COMPACT_KEEP_RECENT_MESSAGES", "8"))
except ValueError:
    COMPACT_KEEP_RECENT_MESSAGES = 8

try:
    MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "2000"))
except ValueError:
    MEMORY_MAX_CHARS = 2000

# Safety bounds for production misconfiguration.
SESSION_TTL_SECONDS = max(300, SESSION_TTL_SECONDS)
MAX_USER_SESSIONS = max(100, MAX_USER_SESSIONS)
LLM_MAX_CONCURRENCY = max(1, min(LLM_MAX_CONCURRENCY, 64))
EXA_TIMEOUT_SECONDS = max(5.0, min(EXA_TIMEOUT_SECONDS, 120.0))
# Compaction threshold must be at least 4 messages and leave room to keep some recent
# turns verbatim; keep-recent must leave at least one Q&A pair to compact.
COMPACT_THRESHOLD_MESSAGES = max(
    4, min(COMPACT_THRESHOLD_MESSAGES, MAX_HISTORY_MESSAGES)
)
COMPACT_KEEP_RECENT_MESSAGES = max(
    2, min(COMPACT_KEEP_RECENT_MESSAGES, COMPACT_THRESHOLD_MESSAGES - 2)
)
MEMORY_MAX_CHARS = max(200, min(MEMORY_MAX_CHARS, 8000))

MAX_MESSAGE_LENGTH = 4096
MAX_INPUT_LENGTH = 4000
MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

_TELEGRAM_FORMAT_RULES = (
    "TELEGRAM FORMATTING RULES (strict):\n"
    "- Plain Telegram Markdown only: *bold*, _italic_, `code`, ```code blocks```, "
    "and [text](https://url) inline links (these render as clickable hyperlinks)\n"
    "- Never use ## headers - they do not render in Telegram\n"
    "- Never use LaTeX math notation\n"
    "- Use flat bullet lists (- item). Never nest lists\n"
    "- Never use | pipe | tables - they do not render in Telegram\n"
    "- For comparisons: use *bold item name* on its own line, then bullet points for attributes\n"
    "- Keep responses concise. Avoid long preambles and sign-offs\n"
    "- When code is requested, use fenced code blocks with the language name"
)

SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant speaking through Telegram.\n\n"
    + _TELEGRAM_FORMAT_RULES
    + "\n\nBEHAVIOUR:\n"
    "- Be direct and practical\n"
    "- If you don't know something, say so rather than guessing"
)

_PROMPT_ORCHESTRATOR = (
    "You are a helpful AI assistant on Telegram with access to a `web_answer` tool "
    "that returns a fresh, web-grounded answer for any factual query.\n\n"
    "Today's date is {date} (current year: {year}).\n\n"
    "WHEN TO CALL `web_answer`:\n"
    "- Current events, news, recent developments\n"
    "- Time-sensitive data (prices, weather, sports scores, stock values)\n"
    "- Latest versions, updates, releases, recommendations ({year}-specific)\n"
    "- Specific facts you are uncertain about or that may have changed recently\n"
    "- Anything that requires fresh information from the web\n\n"
    "WHEN TO ANSWER DIRECTLY (no tool call):\n"
    "- Greetings, small talk, acknowledgments (hi, thanks, ok, etc.)\n"
    "- Questions about your own capabilities or how you work\n"
    "- Well-established facts (capitals, basic math, common definitions)\n"
    "- General programming help with established languages\n"
    "- Concept explanations and how-to questions independent of current tools\n"
    "- Follow-up questions fully answerable from prior conversation\n\n"
    "MANDATORY QUERY-REWRITING RULES (when calling `web_answer`):\n"
    "The `query` argument MUST be fully self-contained because the search engine has NO memory "
    "of this conversation.\n"
    "- Resolve every pronoun (it, they, this, that, etc.) using prior turns\n"
    "- Re-inject all relevant entities mentioned earlier (model names, products, people, places)\n"
    "- For time-sensitive topics, include the current year {year}\n"
    "- Keep the query concise (5-15 words). Do not wrap in quotes unless the phrase is an exact name\n"
    "- Do NOT just echo the user's message verbatim if it contains unresolved references\n\n"
    "Examples of good rewriting:\n"
    "  Prior turn: 'tell me about the M5 MacBook Pro'\n"
    "  User now: 'how does it compare to the M4?'\n"
    "  query: M5 MacBook Pro vs M4 MacBook Pro comparison {year}\n\n"
    "  User now: 'and the price?'\n"
    "  query: M5 MacBook Pro price {year}\n\n"
    "TOOL USAGE LIMITS:\n"
    "- Use `web_answer` AT MOST ONCE per user turn. Pick the single most useful query.\n"
    "- Never call the tool for greetings or self-capability questions.\n\n"
    "WHEN ANSWERING DIRECTLY, follow these rules:\n"
    + _TELEGRAM_FORMAT_RULES
)

_PROMPT_FORMATTER = (
    "You are a helpful AI assistant on Telegram. The user asked a question and a fresh, "
    "web-grounded answer was retrieved for you. It appears appended to the user's latest "
    "message after a '--- Web research' marker.\n\n"
    "Today's date is {date}.\n\n"
    "YOUR TASK:\n"
    "- Rewrite the web research as a clear, direct answer to the user's question.\n"
    "- Preserve ALL factual content from the research. Do NOT invent facts that are not in it.\n"
    "- If the research is missing information needed to answer, say so honestly.\n"
    "- Do NOT mention 'the search results' or 'the web answer' explicitly - speak naturally.\n"
    "- The research often contains inline source links in the form [Site Name](https://url). "
    "PRESERVE these links exactly - keep them where they appear. They render as clickable "
    "hyperlinks in Telegram and serve as sources for the user. Do NOT strip or rewrite them.\n"
    "- DO strip any numeric citation markers like [1], [2], (1, 2), etc. if they ever appear.\n"
    "- If the research says it failed or returned nothing, apologize briefly and offer to retry.\n\n"
    + _TELEGRAM_FORMAT_RULES
)


_PROMPT_SUMMARIZER = (
    "You are a conversation summarizer for an AI Telegram assistant.\n\n"
    "Goal: produce a concise running memory (at most {max_chars} characters) that captures "
    "everything the assistant will need to handle follow-up questions once the earliest "
    "turns of this chat fall out of the verbatim context window.\n\n"
    "MUST PRESERVE:\n"
    "- Topics and entities discussed (products, names, places, projects, files, code)\n"
    "- User identity hints, stated preferences, constraints, goals\n"
    "- Factual conclusions, decisions made, recommendations given\n"
    "- Outstanding questions or open threads\n\n"
    "FORMAT:\n"
    "- Plain prose, optionally with short bullets\n"
    "- Third person from the assistant's POV (e.g. 'The user asked about X; I explained Y.')\n"
    "- No greetings, acknowledgments, or filler\n"
    "- No Telegram markdown - this memory is internal, not user-facing\n\n"
    "OUTPUT: ONLY the new combined summary, no preamble or commentary."
)


def _make_orchestrator_prompt() -> str:
    now = datetime.datetime.now()
    return _PROMPT_ORCHESTRATOR.format(
        date=now.strftime("%Y-%m-%d"), year=now.strftime("%Y")
    )


def _make_formatter_prompt() -> str:
    now = datetime.datetime.now()
    return _PROMPT_FORMATTER.format(date=now.strftime("%Y-%m-%d"))


def _build_system_with_memory(base_prompt: str, memory: Optional[str]) -> str:
    """Append a compact memory section to a base system prompt.

    The memory is a summary of conversation turns that have fallen out of the
    verbatim history window. It is treated as background knowledge of the
    conversation, separate from the recent turn-by-turn messages.
    """
    if not memory or not memory.strip():
        return base_prompt
    return (
        base_prompt
        + "\n\n## Conversation memory (earlier turns, summarized)\n"
        + memory.strip()
    )


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by the model when thinking is disabled."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


# Match only NUMERIC citation markers: [1], [2], [1, 2]. The negative lookahead
# `(?!\()` ensures we never touch `[Title](url)` markdown links - Exa embeds
# clickable sources in that form and we want Telegram to render them.
_CITATION_MARKER_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\](?!\()")


def _strip_citation_markers(text: str) -> str:
    """Remove numeric citation markers like [1], [2], [1, 2] from Exa answers.

    Preserves markdown source links of the form `[Title](url)` so Telegram can
    render them as clickable hyperlinks.
    """
    return _CITATION_MARKER_RE.sub("", text or "").strip()


@tool
def web_answer(query: str) -> str:
    """Search the web and return a fresh, grounded answer with citations.

    Use this for current events, recent releases, time-sensitive data (prices, weather,
    sports scores, stocks), latest versions, news, or any factual question where you need
    fresh web information. Do NOT use this for greetings, small talk, self-capability
    questions, well-established facts, general programming help, or concept explanations.

    The query MUST be fully self-contained: resolve all pronouns and re-inject every
    relevant entity from prior conversation, and include the current year for
    time-sensitive topics. The search engine has no memory of this chat.

    Args:
        query: A self-contained search query (5-15 words). Example:
            "M5 MacBook Pro vs M4 MacBook Pro performance comparison 2026"
    """
    raise NotImplementedError("web_answer is executed by the agent loop, not LangChain")


_TRANSIENT_ERROR_KEYWORDS = (
    "rate limit",
    "too many requests",
    "429",
    "timeout",
    "timed out",
    "503",
    "502",
    "500",
    "529",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "connection error",
    "connection reset",
)
_CHAT_MAX_RETRIES = 2
_CHAT_RETRY_BASE_DELAY = 3.0
_CHAT_ATTEMPT_TIMEOUT = 120.0


class NvidiaLangChainProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _make_client(self, enable_thinking: bool, max_tokens: int) -> ChatNVIDIA:
        return ChatNVIDIA(
            model=MODEL_ID,
            api_key=self.api_key,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_completion_tokens=max_tokens,
            model_kwargs={
                "reasoning_budget": REASONING_BUDGET,
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            },
        )

    def chat(
        self,
        messages: List[Dict],
        enable_thinking: bool = True,
        show_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        max_out = max_tokens or MAX_TOKENS
        chat_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        client = self._make_client(enable_thinking=enable_thinking, max_tokens=max_out)

        reasoning_parts: List[str] = []
        content_parts: List[str] = []
        for chunk in client.stream(chat_messages):
            if getattr(chunk, "content", None):
                content_parts.append(chunk.content)

            add_kw = getattr(chunk, "additional_kwargs", None)
            if add_kw and isinstance(add_kw, dict):
                reasoning = add_kw.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(str(reasoning))

        if not content_parts and not reasoning_parts:
            raise ValueError("API returned empty response.")

        if show_thinking and reasoning_parts:
            return (
                "💭 *Thinking:*\n"
                + "".join(reasoning_parts)
                + "\n\n"
                + "".join(content_parts)
            )

        cleaned = _strip_think_tags("".join(content_parts))
        if not cleaned:
            # Some short formatter calls put everything inside <think>...</think>.
            # Extract the inner block as a fallback so we still return something.
            m_think = re.search(
                r"<think>([\s\S]*?)</think>", "".join(content_parts), re.IGNORECASE
            )
            if m_think:
                cleaned = m_think.group(1).strip()
        if not cleaned and reasoning_parts:
            cleaned = "".join(reasoning_parts).strip()
        return cleaned

    def decide(
        self,
        messages: List[Dict],
        tools: List,
        max_tokens: Optional[int] = None,
    ):
        """Run a single non-streaming routing call with bound tools.

        Returns the raw AIMessage so callers can inspect both `.content` (direct
        answer text) and `.tool_calls` (structured tool invocations).
        """
        max_out = max_tokens or 1024
        client = self._make_client(enable_thinking=False, max_tokens=max_out)
        client_with_tools = client.bind_tools(list(tools))
        return client_with_tools.invoke(messages)


class ExaAnswerService:
    """Calls Exa's /answer endpoint with multi-key rotation and retry semantics."""

    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self._idx = 0
        self._lock = threading.Lock()

    def _next_index(self) -> int:
        with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self.api_keys)
            return idx

    @staticmethod
    def _extract_answer(result) -> str:
        """Pull the answer text out of an exa_py AnswerResponse."""
        answer = getattr(result, "answer", None)
        if isinstance(answer, str):
            return answer.strip()
        if isinstance(result, dict):
            return str(result.get("answer") or "").strip()
        return ""

    def answer_sync(self, query: str) -> Optional[str]:
        """Return a citation-stripped answer string, or None on total failure.

        Rotates through all keys on any failure (transient or permanent),
        because permanent failures are usually key-specific (revoked/invalid)
        and another key may still succeed.
        """
        if not self.api_keys:
            return None

        start_idx = self._next_index()
        last_err: Optional[Exception] = None

        for offset in range(len(self.api_keys)):
            key_idx = (start_idx + offset) % len(self.api_keys)
            api_key = self.api_keys[key_idx]
            try:
                exa = Exa(api_key)
                kwargs: Dict = {}
                if EXA_ANSWER_MODEL:
                    kwargs["model"] = EXA_ANSWER_MODEL
                result = exa.answer(query, **kwargs)

                raw_answer = self._extract_answer(result)
                if not raw_answer:
                    raise ValueError("Exa returned empty answer")

                clean = _strip_citation_markers(raw_answer)
                citation_count = len(getattr(result, "citations", None) or [])

                logger.info(
                    "[Answer] Exa key=%s query='%s' answer_len=%s citations=%s",
                    key_idx + 1,
                    query,
                    len(clean),
                    citation_count,
                )
                return clean
            except Exception as e:
                last_err = e
                logger.warning("[Answer] Exa key=%s failed: %s", key_idx + 1, e)
                continue

        if last_err:
            logger.error("[Answer] Exa failed for query '%s': %s", query, last_err)
        return None


provider = NvidiaLangChainProvider(NVIDIA_API_KEY or "")
answer_service = ExaAnswerService(EXA_API_KEYS)
user_sessions: Dict[str, Dict] = {}
_llm_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazily create the LLM semaphore inside the running event loop.

    Creating asyncio.Semaphore at module level before a loop exists is
    deprecated in Python 3.10+ and raises a DeprecationWarning in 3.12+.
    """
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
    return _llm_semaphore


def _prune_user_sessions(now: float):
    stale_ids = [
        uid
        for uid, sess in user_sessions.items()
        if now - sess.get("last_seen", now) > SESSION_TTL_SECONDS
    ]
    for uid in stale_ids:
        user_sessions.pop(uid, None)

    if len(user_sessions) <= MAX_USER_SESSIONS:
        return

    # Drop oldest sessions until under cap.
    excess = len(user_sessions) - MAX_USER_SESSIONS
    oldest = sorted(user_sessions.items(), key=lambda kv: kv[1].get("last_seen", now))[
        :excess
    ]
    for uid, _ in oldest:
        user_sessions.pop(uid, None)


def get_user_session(user_id: str) -> Dict:
    now = time.time()
    _prune_user_sessions(now)
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "memory": "",
            "thinking_enabled": False,
            "web_search": True,
            "last_seen": now,
            "cancel_event": asyncio.Event(),
            "compact_in_progress": False,
            "_bg_tasks": set(),
        }
    else:
        sess = user_sessions[user_id]
        sess["last_seen"] = now
        # Backfill defaults for sessions created before compaction shipped.
        sess.setdefault("memory", "")
        sess.setdefault("compact_in_progress", False)
        sess.setdefault("_bg_tasks", set())
    return user_sessions[user_id]


def is_user_allowed(user_id: str) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def _trim_history(history: list) -> list:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history

    excess = len(history) - MAX_HISTORY_MESSAGES
    pairs_to_remove = (excess + 1) // 2
    messages_to_remove = pairs_to_remove * 2
    del history[:messages_to_remove]
    return history


async def reply_text_safe(message, text: str):
    try:
        await message.reply_text(text, parse_mode="Markdown")
    except Exception:
        try:
            await message.reply_text(text)
        except Exception as e:
            logger.error("Failed to send message: %s", e)
            raise


async def _edit_text_safe(placeholder, text: str) -> bool:
    """Try to edit a sent message in place. Returns True on success.

    Falls back from Markdown to plain text on parser errors, mirroring
    `reply_text_safe`. Returns False if the message can no longer be edited
    (deleted, too old, etc.) so callers can fall back to sending a new message.
    """
    try:
        await placeholder.edit_text(text, parse_mode="Markdown")
        return True
    except Exception:
        pass
    try:
        await placeholder.edit_text(text)
        return True
    except Exception as e:
        logger.warning("[Bot] Failed to edit placeholder, will send new: %s", e)
        return False


def _split_into_chunks(text: str, chunk_limit: int) -> List[str]:
    """Split `text` into Telegram-safe chunks at line/word boundaries."""
    chunks: List[str] = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(line) > chunk_limit:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            remaining = line
            while remaining:
                if len(remaining) <= chunk_limit:
                    chunks.append(remaining)
                    break
                split_at = remaining.rfind(" ", 0, chunk_limit)
                if split_at <= 0:
                    split_at = chunk_limit
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at:].lstrip()
            continue
        if len(current_chunk) + len(line) + 1 > chunk_limit:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = (current_chunk + "\n" + line) if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


async def _deliver_response(message, placeholder, response_text: str) -> None:
    """Send `response_text` to the user, editing `placeholder` in place when possible.

    - If `placeholder` is None, behaves like `reply_text_safe` (with chunking).
    - If `placeholder` is a Message, the first chunk replaces its text via `edit_text`;
      additional chunks are sent as new messages.
    - On any edit failure, falls back to sending the chunk as a new reply.
    """
    if len(response_text) <= MAX_MESSAGE_LENGTH:
        if placeholder is not None and await _edit_text_safe(placeholder, response_text):
            return
        await reply_text_safe(message, response_text)
        return

    header_reserve = 25
    chunk_limit = MAX_MESSAGE_LENGTH - header_reserve
    chunks = _split_into_chunks(response_text, chunk_limit)
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        header = f"\U0001f4c4 Part {i}/{total}\n\n" if total > 1 else ""
        payload = header + chunk
        if i == 1 and placeholder is not None:
            if await _edit_text_safe(placeholder, payload):
                continue
        await reply_text_safe(message, payload)


async def _call_with_retry(
    factory: Callable[[], Any],
    *,
    label: str,
    cancel_event: Optional["asyncio.Event"] = None,
) -> Any:
    """Generic semaphore-gated retry wrapper for sync LLM calls.

    `factory` is a zero-arg callable already bound with all keyword args.
    `label` is a short tag used in retry warning logs.
    """
    e_for_delay: BaseException = RuntimeError("uninitialized")
    for attempt in range(1, _CHAT_MAX_RETRIES + 2):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Cancelled by /restart")
        try:
            async with _get_semaphore():
                return await asyncio.wait_for(
                    asyncio.to_thread(factory),
                    timeout=_CHAT_ATTEMPT_TIMEOUT,
                )
        except asyncio.TimeoutError as e:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            e_for_delay = e
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            if not any(kw in str(e).lower() for kw in _TRANSIENT_ERROR_KEYWORDS):
                raise
            e_for_delay = e

        delay = _CHAT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
        logger.warning(
            "[Bot] %s transient error attempt %s/%s: %s. Retry in %ss",
            label,
            attempt,
            _CHAT_MAX_RETRIES + 1,
            e_for_delay,
            int(delay),
        )
        if cancel_event:
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=delay)
                raise asyncio.CancelledError("Cancelled by /restart")
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(delay)
    # Unreachable: every iteration either returns or raises above.
    raise RuntimeError(f"{label}: exhausted retries without raising")


async def _chat_with_retry(
    provider_obj,
    messages: list,
    enable_thinking: bool = True,
    show_thinking: bool = False,
    max_tokens: Optional[int] = None,
    cancel_event: Optional["asyncio.Event"] = None,
) -> str:
    def _do() -> str:
        return provider_obj.chat(
            messages=messages,
            enable_thinking=enable_thinking,
            show_thinking=show_thinking,
            max_tokens=max_tokens or MAX_TOKENS,
        )

    result = await _call_with_retry(_do, label="API", cancel_event=cancel_event)
    return result or ""


async def _decide_with_retry(
    provider_obj,
    messages: list,
    tools: list,
    max_tokens: Optional[int] = None,
    cancel_event: Optional["asyncio.Event"] = None,
):
    """Orchestrator call (bind_tools + invoke) with transient-error retries.

    Returns the AIMessage produced by LangChain so the caller can inspect both
    `.content` (direct answer) and `.tool_calls` (structured tool invocations).
    """

    def _do():
        return provider_obj.decide(
            messages=messages, tools=tools, max_tokens=max_tokens
        )

    return await _call_with_retry(
        _do, label="Orchestrator", cancel_event=cancel_event
    )


async def _summarize_history(
    provider_obj,
    existing_memory: str,
    old_messages: List[Dict],
) -> str:
    """Fold a chunk of old messages into a single running-memory summary.

    Returns the new combined summary (truncated to MEMORY_MAX_CHARS). On any
    failure, returns the existing memory unchanged so the caller can decide
    whether to mutate session state.
    """
    if not old_messages:
        return existing_memory or ""

    lines: List[str] = []
    for m in old_messages:
        role = (m.get("role") or "?").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Truncate very long single messages so the summarizer prompt stays bounded.
        if len(content) > 1200:
            content = content[:1200] + "...(truncated)"
        lines.append(f"{role}: {content}")

    sections: List[str] = []
    if existing_memory and existing_memory.strip():
        sections.append("EXISTING SUMMARY:\n" + existing_memory.strip())
    if lines:
        sections.append("NEW MESSAGES TO INCORPORATE:\n" + "\n\n".join(lines))
    if not sections:
        return existing_memory or ""

    summarize_messages = [
        {
            "role": "system",
            "content": _PROMPT_SUMMARIZER.format(max_chars=MEMORY_MAX_CHARS),
        },
        {"role": "user", "content": "\n\n".join(sections)},
    ]

    try:
        summary = await _chat_with_retry(
            provider_obj=provider_obj,
            messages=summarize_messages,
            enable_thinking=False,
            show_thinking=False,
            max_tokens=max(512, MEMORY_MAX_CHARS // 2),
        )
    except Exception as e:
        logger.warning("[Compact] summarize call failed: %s", e)
        return existing_memory or ""

    summary = (summary or "").strip()
    if not summary:
        return existing_memory or ""
    if len(summary) > MEMORY_MAX_CHARS:
        # Trim on a sentence boundary when possible to avoid cutting mid-word.
        head = summary[:MEMORY_MAX_CHARS]
        cut = head.rfind(". ")
        summary = (head[: cut + 1] if cut > MEMORY_MAX_CHARS // 2 else head).strip()
    return summary


async def _maybe_compact_session(provider_obj, session: Dict) -> None:
    """Background task: fold old turns into `session['memory']` when history exceeds threshold.

    Safe to call frequently - returns immediately if compaction is disabled,
    already in progress, or not yet needed. On completion, atomically removes
    only the prefix that was actually summarized, so concurrent appends or
    `/clear` operations during summarization do not corrupt history.
    """
    if not COMPACT_ENABLED:
        return
    if session.get("compact_in_progress"):
        return

    history = session.get("history")
    if not isinstance(history, list):
        return

    convo_count = sum(
        1 for m in history if m.get("role") in ("user", "assistant")
    )
    if convo_count <= COMPACT_THRESHOLD_MESSAGES:
        return

    keep = COMPACT_KEEP_RECENT_MESSAGES
    if len(history) <= keep:
        return

    session["compact_in_progress"] = True
    try:
        # Snapshot the prefix we intend to summarize. After the await below
        # the history may have grown (new turns) or been reset (/clear).
        old_messages = list(history[:-keep])
        if not old_messages:
            return
        existing_memory = session.get("memory") or ""

        logger.info(
            "[Compact] start: history=%s memory_chars=%s prefix_to_summarize=%s",
            len(history),
            len(existing_memory),
            len(old_messages),
        )

        new_summary = await _summarize_history(
            provider_obj=provider_obj,
            existing_memory=existing_memory,
            old_messages=old_messages,
        )
        if not new_summary or not new_summary.strip():
            logger.warning("[Compact] empty summary; skipping mutation")
            return

        # Atomic safety check before mutating: the list reference must still be
        # the same object AND its first message must match what we summarized.
        # If `/clear` ran during summarization, the reference changes; if the
        # prefix changed for some other reason, we abort to stay consistent.
        current = session.get("history")
        if (
            current is not history
            or len(current) < len(old_messages)
            or current[0] != old_messages[0]
        ):
            logger.info("[Compact] aborted: history mutated during summarize")
            return

        del history[: len(old_messages)]
        session["memory"] = new_summary
        logger.info(
            "[Compact] done: removed=%s remaining_history=%s memory_chars=%s",
            len(old_messages),
            len(history),
            len(new_summary),
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("[Compact] failed: %s", e)
    finally:
        session["compact_in_progress"] = False


def _spawn_background(coro, bag: set) -> asyncio.Task:
    """Fire-and-forget an awaitable, keeping a strong reference until it completes.

    asyncio only holds weak references to tasks, so without this helper a
    background task can be garbage-collected mid-flight.
    """
    task = asyncio.create_task(coro)
    bag.add(task)
    task.add_done_callback(bag.discard)
    return task


async def _exa_answer(
    query: str, cancel_event: Optional["asyncio.Event"] = None
) -> Optional[str]:
    """Async wrapper around Exa /answer with timeout, multi-key rotation, and cancel.

    Races the underlying sync call against `cancel_event` so `/restart` aborts the
    wait promptly. Note: cancelling the awaitable does NOT kill the underlying
    thread (it completes in the background and its result is discarded).
    """
    answer_task = asyncio.create_task(
        asyncio.to_thread(answer_service.answer_sync, query)
    )

    if cancel_event is None:
        try:
            return await asyncio.wait_for(answer_task, timeout=EXA_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("[Answer] Exa /answer timed out for '%s'", query)
            return None
        except Exception as e:
            logger.warning("[Answer] Exa /answer failed for '%s': %s", query, e)
            return None

    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {answer_task, cancel_task},
            timeout=EXA_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done:
            raise asyncio.CancelledError("Cancelled by /restart")
        if answer_task in done:
            try:
                return answer_task.result()
            except Exception as e:
                logger.warning(
                    "[Answer] Exa /answer failed for '%s': %s", query, e
                )
                return None
        # Neither completed within the timeout.
        logger.warning("[Answer] Exa /answer timed out for '%s'", query)
        return None
    finally:
        for t in (answer_task, cancel_task):
            if not t.done():
                t.cancel()


def _extract_tool_query(tool_call) -> Optional[str]:
    """Pull the `query` arg out of a LangChain tool_call dict robustly.

    LangChain normalizes tool_calls to {"name": "web_answer", "args": {...}, "id": "..."}.
    We also accept the raw OpenAI shape {"function": {"arguments": "<json string>"}} as
    a defensive fallback for unusual provider configurations.
    """
    args: Any = None
    if isinstance(tool_call, dict):
        args = tool_call.get("args")
        if args is None and "function" in tool_call:
            fn = tool_call.get("function") or {}
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = None
            elif isinstance(raw, dict):
                args = raw
    if not isinstance(args, dict):
        return None
    q = args.get("query")
    if not isinstance(q, str):
        return None
    q = q.strip()
    return q or None


async def _agent_answer_loop(
    provider_obj,
    session_history: list,
    user_message: str,
    thinking_enabled: bool,
    cancel_event: Optional["asyncio.Event"] = None,
    on_tool_call: Optional[Callable[[], Awaitable[Any]]] = None,
    session_memory: Optional[str] = None,
) -> str:
    """Single-turn tool-calling agent loop.

    FLOW:
      1. Orchestrator: NVIDIA LLM (with `web_answer` bound) sees full history plus
         the running `session_memory` (summary of older turns) and either:
           (a) answers the user directly -> return that content, OR
           (b) emits a tool_call with a self-contained rewritten query.
      2. If tool_call: run Exa /answer, then call a formatter LLM pass that rewrites
         the grounded answer in Telegram markdown using the user's latest message
         as the natural anchor.
      3. If Exa fails: tell the formatter so it can apologize cleanly.
    """
    orchestrator_system = _build_system_with_memory(
        _make_orchestrator_prompt(), session_memory
    )
    orchestrator_messages = (
        [{"role": "system", "content": orchestrator_system}]
        + list(session_history)
    )

    ai_msg = await _decide_with_retry(
        provider_obj=provider_obj,
        messages=orchestrator_messages,
        tools=[web_answer],
        cancel_event=cancel_event,
    )

    tool_calls = list(getattr(ai_msg, "tool_calls", None) or [])
    direct_content = _strip_think_tags(str(getattr(ai_msg, "content", "") or ""))

    if not tool_calls:
        if direct_content:
            logger.info("[Agent] orchestrator answered directly (no tool call)")
            return direct_content
        # Empty content AND no tool calls -> fall back to a normal chat call.
        logger.warning("[Agent] orchestrator returned empty; falling back to chat")
        fallback_system = _build_system_with_memory(SYSTEM_PROMPT, session_memory)
        return await _chat_with_retry(
            provider_obj=provider_obj,
            messages=[{"role": "system", "content": fallback_system}]
            + list(session_history),
            enable_thinking=True,
            show_thinking=thinking_enabled,
            cancel_event=cancel_event,
        )

    tc = tool_calls[0]
    query = _extract_tool_query(tc) or user_message[:120].strip() or user_message
    logger.info("[Agent] tool=web_answer query='%s'", query)

    if on_tool_call is not None:
        try:
            await on_tool_call()
        except Exception:
            # Notice failures (Telegram blip) must never abort the actual research.
            pass

    answer_text = await _exa_answer(query, cancel_event=cancel_event)
    if not answer_text:
        research_block = (
            "(Web search failed or returned no useful results. "
            "Apologize briefly to the user and suggest they retry.)"
        )
    else:
        research_block = answer_text

    formatter_history = [
        m for m in session_history if m.get("role") != "system"
    ]
    if formatter_history and formatter_history[-1].get("role") == "user":
        last = formatter_history[-1]
        augmented_user = (
            f"{last.get('content', '')}\n\n"
            f"--- Web research (query: \"{query}\") ---\n"
            f"{research_block}\n"
            f"--- end research ---"
        )
        formatter_history[-1] = {"role": "user", "content": augmented_user}
    else:
        formatter_history.append(
            {
                "role": "user",
                "content": (
                    f"{user_message}\n\n"
                    f"--- Web research (query: \"{query}\") ---\n"
                    f"{research_block}\n"
                    f"--- end research ---"
                ),
            }
        )

    formatter_system = _build_system_with_memory(
        _make_formatter_prompt(), session_memory
    )
    formatter_messages = (
        [{"role": "system", "content": formatter_system}] + formatter_history
    )

    return await _chat_with_retry(
        provider_obj=provider_obj,
        messages=formatter_messages,
        enable_thinking=True,
        show_thinking=thinking_enabled,
        cancel_event=cancel_event,
    )


async def _keep_typing(chat) -> None:
    """Re-send the 'typing' action every 4 s so the indicator stays alive during long requests."""
    try:
        while True:
            await chat.send_action(action="typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _clean_for_history(response: str) -> str:
    """Strip the thinking-trace header before storing a response in conversation history.

    The full formatted response (with the 💭 header) is sent to Telegram, but the
    clean answer is what gets stored so subsequent turns don't have formatting noise
    in the context window.
    """
    if response.startswith("\U0001f4ad *Thinking:*\n"):
        sep = response.find("\n\n", len("\U0001f4ad *Thinking:*\n"))
        if sep != -1:
            return response[sep + 2 :].strip()
    return response


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "\u26d4 Sorry, you're not authorized to use this bot."
        )
        return

    session = get_user_session(user_id)
    web_status = "ON \u2705" if session.get("web_search", True) else "OFF \U0001f515"
    think_status = (
        "ON \u2705" if session.get("thinking_enabled", False) else "OFF \U0001f515"
    )
    history_len = len(
        [m for m in session.get("history", []) if m.get("role") != "system"]
    )
    await update.message.reply_text(
        "\U0001f916 *NVIDIA + Exa Assistant*\n\n"
        f"\U0001f9e0 Model: `{MODEL_ID}`\n"
        f"\U0001f310 Web Access: {web_status}\n"
        f"\U0001f4ad Thinking: {think_status}\n"
        f"\U0001f4dc History: {history_len} messages\n\n"
        "*Commands:*\n"
        "`/status` \u2014 show current settings\n"
        "`/web on|off` \u2014 toggle web access (Exa /answer)\n"
        "`/thinking on|off` \u2014 toggle reasoning traces\n"
        "`/clear` \u2014 clear conversation history\n"
        "`/restart` \u2014 cancel a stuck request\n"
        "`/help` \u2014 full help",
        parse_mode="Markdown",
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    session["history"] = []
    session["memory"] = ""
    await update.message.reply_text(
        "🗑️ Conversation history and memory cleared!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    await update.message.reply_text(
        "💡 *How to use:*\n\n"
        "Send a message. The model decides whether to answer directly or call Exa's "
        "web-grounded `/answer` endpoint and reformats the result for Telegram.\n"
        "Web access is ON by default.\n\n"
        "*Commands:*\n"
        "• `/web` - show web access status\n"
        "• `/web on` / `/web off` - toggle Exa /answer tool\n"
        "• `/thinking on` / `/thinking off` - reasoning traces\n"
        "• `/clear` - clear conversation history\n"
        "• `/restart` - cancel pending request\n"
        "• `/help` - this message",
        parse_mode="Markdown",
    )


async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return

    session = get_user_session(user_id)
    if not context.args:
        state = "enabled" if session.get("thinking_enabled", False) else "disabled"
        await update.message.reply_text(
            f"💭 *Thinking Mode:* {state}\n\nUse `/thinking on` or `/thinking off`.",
            parse_mode="Markdown",
        )
        return

    arg = context.args[0].lower()
    if arg == "on":
        session["thinking_enabled"] = True
        await update.message.reply_text(
            "✅ *Thinking mode enabled.*", parse_mode="Markdown"
        )
    elif arg == "off":
        session["thinking_enabled"] = False
        await update.message.reply_text(
            "🔕 *Thinking mode disabled.*", parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "❌ Use `/thinking on` or `/thinking off`", parse_mode="Markdown"
        )


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return

    session = get_user_session(user_id)
    if not context.args:
        status = "ON" if session.get("web_search") else "OFF"
        await update.message.reply_text(
            f"🌐 *Web Access:* {status}\n"
            "🔎 *Engine:* Exa `/answer` (grounded)\n\n"
            "Use: `/web on` or `/web off`",
            parse_mode="Markdown",
        )
        return

    arg = context.args[0].lower()
    if arg == "on":
        session["web_search"] = True
        await update.message.reply_text("✅ Web access enabled (Exa /answer).")
    elif arg == "off":
        session["web_search"] = False
        await update.message.reply_text("🔕 Web access disabled.")
    else:
        await update.message.reply_text("❌ Use: `/web on|off`", parse_mode="Markdown")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return

    session = get_user_session(user_id)
    cancel_event: asyncio.Event = session["cancel_event"]
    cancel_event.set()

    history = session["history"]
    if history and history[-1].get("role") == "user":
        history.pop()

    await update.message.reply_text(
        "\U0001f6d1 *Cancellation requested.*\n\n"
        "Pending NVIDIA / Exa calls will abort at the next checkpoint. "
        "You can send a new message now.",
        parse_mode="Markdown",
    )
    # Clear the event so the next incoming message is not immediately cancelled.
    cancel_event.clear()


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "\u26d4 Sorry, you're not authorized to use this bot."
        )
        return

    session = get_user_session(user_id)
    web = "ON \u2705" if session.get("web_search", True) else "OFF \U0001f515"
    thinking = (
        "ON \u2705" if session.get("thinking_enabled", False) else "OFF \U0001f515"
    )
    history_len = len(
        [m for m in session.get("history", []) if m.get("role") != "system"]
    )
    exa_model_label = EXA_ANSWER_MODEL or "default"
    memory_chars = len((session.get("memory") or "").strip())
    if not COMPACT_ENABLED:
        memory_status = "disabled"
    elif memory_chars:
        memory_status = f"{memory_chars} chars"
    else:
        memory_status = "empty"
    await update.message.reply_text(
        "\U0001f4ca *Current Settings*\n\n"
        f"\U0001f9e0 *Model:* `{MODEL_ID}`\n"
        f"\U0001f4ad *Thinking:* {thinking}\n"
        f"\U0001f310 *Web Access:* {web}\n"
        f"\U0001f50e *Exa model:* `{exa_model_label}`\n"
        f"\U0001f4dc *History:* {history_len} messages\n"
        f"\U0001f9e9 *Memory:* {memory_status}\n"
        f"\U0001f511 *Exa keys:* {len(EXA_API_KEYS)}",
        parse_mode="Markdown",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return

    user_message = update.message.text
    if not user_message:
        return
    if len(user_message) > MAX_INPUT_LENGTH:
        await update.message.reply_text(
            f"❌ Message too long ({len(user_message):,} chars). "
            f"Please keep it under {MAX_INPUT_LENGTH:,} characters."
        )
        return

    session = get_user_session(user_id)
    cancel_event: asyncio.Event = session["cancel_event"]
    cancel_event.clear()
    assistant_appended = False
    typing_task: Optional[asyncio.Task] = None
    # Captured by the `_notify_searching` closure so we can edit it in-place later.
    placeholder_holder: Dict[str, Any] = {"msg": None}

    try:
        if cancel_event.is_set():
            return

        typing_task = asyncio.create_task(_keep_typing(update.message.chat))

        thinking_enabled = session.get("thinking_enabled", False)
        web_on = session.get("web_search", True)
        session_memory = session.get("memory", "")

        session["history"].append({"role": "user", "content": user_message})

        if not web_on:
            logger.info("[Bot] user=%s direct path (web off)", user_id)
            direct_messages = [
                {
                    "role": "system",
                    "content": _build_system_with_memory(
                        SYSTEM_PROMPT, session_memory
                    ),
                }
            ] + list(session["history"])
            bot_response = await _chat_with_retry(
                provider_obj=provider,
                messages=direct_messages,
                enable_thinking=True,
                show_thinking=thinking_enabled,
                cancel_event=cancel_event,
            )
        else:
            logger.info("[Bot] user=%s agent loop (web on)", user_id)

            async def _notify_searching() -> None:
                try:
                    placeholder_holder["msg"] = await update.message.reply_text(
                        "\U0001f50d Searching the web\u2026"
                    )
                except Exception:
                    pass

            bot_response = await _agent_answer_loop(
                provider_obj=provider,
                session_history=session["history"],
                user_message=user_message,
                thinking_enabled=thinking_enabled,
                cancel_event=cancel_event,
                on_tool_call=_notify_searching,
                session_memory=session_memory,
            )

        if not bot_response.strip():
            bot_response = (
                "\u26a0\ufe0f The AI returned an empty response. Please try again."
            )

        session["history"].append(
            {"role": "assistant", "content": _clean_for_history(bot_response)}
        )
        assistant_appended = True
        session["history"] = _trim_history(session["history"])

        await _deliver_response(
            update.message, placeholder_holder.get("msg"), bot_response
        )

        # Fire-and-forget background compaction. Returns immediately if the
        # threshold has not been crossed, compaction is already running, or
        # the feature is disabled via env var.
        if COMPACT_ENABLED:
            _spawn_background(
                _maybe_compact_session(provider, session),
                session["_bg_tasks"],
            )

    except asyncio.CancelledError:
        logger.info("[Bot] user=%s request cancelled", user_id)
        if (
            assistant_appended
            and session["history"]
            and session["history"][-1].get("role") == "assistant"
        ):
            session["history"].pop()
        if session["history"] and session["history"][-1].get("role") == "user":
            session["history"].pop()
        placeholder = placeholder_holder.get("msg")
        if placeholder is not None:
            await _edit_text_safe(placeholder, "\U0001f6d1 Cancelled.")
    except Exception as e:
        logger.error("Error in handle_message: %s", e, exc_info=True)
        if (
            assistant_appended
            and session["history"]
            and session["history"][-1].get("role") == "assistant"
        ):
            session["history"].pop()
        if session["history"] and session["history"][-1].get("role") == "user":
            session["history"].pop()
        error_text = (
            "\u274c Something went wrong processing your request.\n\n"
            "Try:\n\u2022 `/clear` to reset conversation\n"
            "\u2022 `/restart` to cancel stuck calls"
        )
        placeholder = placeholder_holder.get("msg")
        edited = False
        if placeholder is not None:
            edited = await _edit_text_safe(placeholder, error_text)
        if not edited:
            try:
                await update.message.reply_text(error_text, parse_mode="Markdown")
            except Exception:
                pass
    finally:
        if typing_task and not typing_task.done():
            typing_task.cancel()


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required!")
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is required!")
    if not EXA_API_KEYS:
        raise ValueError(
            "EXA_API_KEYS environment variable is required. "
            "Use CSV format like: EXA_API_KEYS=key1,key2,key3"
        )

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("thinking", thinking_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("🚀 NVIDIA + Exa Telegram bot started")
    logger.info("🧠 Model: %s", MODEL_ID)
    logger.info("🔎 Exa keys loaded: %s", len(EXA_API_KEYS))
    logger.info(
        "🌐 Web access default: ON (engine: Exa /answer, model: %s)",
        EXA_ANSWER_MODEL or "default",
    )
    logger.info(
        "⚙️ Agent controls: exa_timeout=%ss llm_concurrency=%s",
        EXA_TIMEOUT_SECONDS,
        LLM_MAX_CONCURRENCY,
    )
    logger.info(
        "🧹 Session controls: ttl_seconds=%s max_sessions=%s",
        SESSION_TTL_SECONDS,
        MAX_USER_SESSIONS,
    )
    if ALLOWED_USER_IDS:
        logger.info(
            "🔐 Access mode: allowlist enabled (%s Telegram UID(s))",
            len(ALLOWED_USER_IDS),
        )
    else:
        logger.info("🔓 Access mode: open (no TELEGRAM_ALLOWED_UIDS set)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
