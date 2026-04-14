import asyncio
import datetime
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from langchain_exa import ExaSearchRetriever
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
EXA_API_KEYS = [k.strip() for k in os.getenv("EXA_API_KEYS", "").split(",") if k.strip()]


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
    EXA_MAX_RESULTS = int(os.getenv("EXA_MAX_RESULTS", "5"))
except ValueError:
    EXA_MAX_RESULTS = 5

try:
    EXA_MAX_SNIPPET_LEN = int(os.getenv("EXA_MAX_SNIPPET_LEN", "500"))
except ValueError:
    EXA_MAX_SNIPPET_LEN = 500

try:
    EXA_TIMEOUT_SECONDS = float(os.getenv("EXA_TIMEOUT_SECONDS", "20"))
except ValueError:
    EXA_TIMEOUT_SECONDS = 20.0

try:
    RESEARCH_MAX_STEPS = int(os.getenv("RESEARCH_MAX_STEPS", "4"))
except ValueError:
    RESEARCH_MAX_STEPS = 4

try:
    RESEARCH_MAX_SNIPPETS = int(os.getenv("RESEARCH_MAX_SNIPPETS", "20"))
except ValueError:
    RESEARCH_MAX_SNIPPETS = 20

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

# Safety bounds for production misconfiguration.
RESEARCH_MAX_STEPS = max(1, min(RESEARCH_MAX_STEPS, 8))
RESEARCH_MAX_SNIPPETS = max(5, min(RESEARCH_MAX_SNIPPETS, 60))
SESSION_TTL_SECONDS = max(300, SESSION_TTL_SECONDS)
MAX_USER_SESSIONS = max(100, MAX_USER_SESSIONS)
LLM_MAX_CONCURRENCY = max(1, min(LLM_MAX_CONCURRENCY, 64))

MAX_MESSAGE_LENGTH = 4096
MAX_INPUT_LENGTH = 4000
MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

_PROMPT_BASE = (
    "You are a helpful, concise AI assistant speaking through Telegram.\n\n"
    "FORMATTING RULES - follow strictly:\n"
    "- Use plain Telegram Markdown only: *bold*, _italic_, `code`, ```code blocks```\n"
    "- Never use ## headers - they do not render in Telegram\n"
    "- Never use LaTeX math notation\n"
    "- Use flat bullet lists (- item). Never nest lists\n"
    "- Keep responses concise. Avoid long preambles and sign-offs\n"
    "- For comparisons: use *bold item name* on its own line, then bullet points for attributes. "
    "Never use | pipe | tables - they do not render in Telegram\n\n"
    "BEHAVIOUR:\n"
    "- Be direct and practical\n"
    "- If you don't know something, say so rather than guessing\n"
    "- When code is requested, use fenced code blocks with the language name"
)

_PROMPT_SEARCH_RESULTS = (
    "\n\nSEARCH RESULTS FORMAT:\n"
    "When web search results are provided, they appear as numbered snippets "
    "(1. Title: text). Base your answer ONLY on information contained in these snippets. "
    "Do NOT use your training data to fill in facts that are absent from the snippets - "
    "if the snippets do not contain enough information to answer confidently, "
    "say so clearly (for example: 'The search results do not specify this'). "
    "You may refer to a result naturally but do NOT write citation numbers like [1]."
)

_PROMPT_RESEARCH_LOOP = (
    "You are an accuracy-first research planner. Today's date is {date} (current year: {year}).\n\n"
    "Task: decide whether to search the web again or return a final answer.\n"
    "You may perform MULTIPLE search rounds before answering.\n\n"
    "Output format (strict):\n"
    "- If more evidence is needed: SEARCH: <query>\n"
    "- If confident enough to answer: FINAL: <answer>\n\n"
    "Rules:\n"
    "- Use SEARCH when information may be time-sensitive or uncertain.\n"
    "- For time-sensitive queries always use the current year ({year}). NEVER use past years like 2024 or 2025.\n"
    "- Refine search queries using what is already known.\n"
    "- Keep query short and specific (4-10 words).\n"
    "- If evidence is sufficient, return FINAL with concise, practical answer.\n"
    "- Do not output anything except SEARCH:... or FINAL:..."
)

SYSTEM_PROMPT = _PROMPT_BASE
SYSTEM_PROMPT_WITH_RESULTS = _PROMPT_BASE + _PROMPT_SEARCH_RESULTS


def _make_research_loop_prompt() -> str:
    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    year = now.strftime("%Y")
    return _PROMPT_RESEARCH_LOOP.format(date=today, year=year)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by the model when thinking is disabled."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


_GREETINGS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hola",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "lol",
        "haha",
        "nice",
        "great",
        "cool",
        "good morning",
        "good night",
        "good evening",
        "good afternoon",
        "ty",
        "thx",
        "np",
        "yw",
    }
)


def _skip_search_decision(text: str) -> bool:
    if len(text) < 6:
        return True
    return text.lower().rstrip("!?.,: ") in _GREETINGS


def _parse_search_query(response: str) -> Optional[str]:
    stripped = response.strip()

    if stripped.upper().startswith("SEARCH:"):
        raw = stripped[7:].strip().split("\n")[0].strip()
    else:
        m = re.search(r"(?i)\bSEARCH:\s*(.+)", stripped)
        if not m:
            return None
        raw = m.group(1).strip().split("\n")[0].strip()

    parts = re.split(r"(?i)NOSEARCH", raw, maxsplit=1)
    if len(parts) == 2 and re.match(r"(?i)\s*SEARCH:\s*", parts[1]):
        raw = re.sub(r"(?i)^SEARCH:\s*", "", parts[1]).strip()
    else:
        raw = parts[0].strip()

    while raw.upper().startswith("SEARCH:"):
        raw = raw[7:].strip()

    m_echo = re.search(r"(?i)SEARCH:", raw)
    if m_echo:
        raw = raw[: m_echo.start()].strip()

    for q in ('"', "'", "`"):
        if len(raw) > 2 and raw[0] == q and raw[-1] == q:
            raw = raw[1:-1].strip()

    raw = raw.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

    m = re.match(r"^[\w\s\-'\".,/&+%@#()]+", raw, re.UNICODE)
    raw = m.group(0).strip() if m else ""

    words = raw.split()
    if len(words) > 10:
        raw = " ".join(words[:10])

    raw = raw[:120]
    return raw if len(raw) >= 2 else None


def _build_search_context(query: str, snippets: list) -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    parts = [f"Today is {today}. Web search results for '{query}':"]
    for i, snip in enumerate(snippets, 1):
        parts.append(f"{i}. {snip}")
    return "\n".join(parts)


def _build_research_context(user_message: str, traces: List[Dict[str, List[str]]]) -> str:
    lines = [f"User question: {user_message}"]
    if not traces:
        lines.append("No previous web searches yet.")
        return "\n".join(lines)

    lines.append("Previous search evidence:")
    snippet_budget = RESEARCH_MAX_SNIPPETS
    for i, t in enumerate(traces, 1):
        lines.append(f"Search {i} query: {t['query']}")
        for j, s in enumerate(t["snippets"], 1):
            if snippet_budget <= 0:
                lines.append("... evidence truncated due to context budget ...")
                return "\n".join(lines)
            lines.append(f"{i}.{j} {s}")
            snippet_budget -= 1
    return "\n".join(lines)


def _parse_research_action(response: str) -> Tuple[str, Optional[str]]:
    stripped = (response or "").strip()
    if not stripped:
        return ("unknown", None)

    if stripped.upper().startswith("SEARCH:"):
        q = _parse_search_query(stripped)
        return ("search", q)

    if stripped.upper().startswith("FINAL:"):
        ans = stripped[6:].strip()
        return ("final", ans if ans else None)

    m_search = re.search(r"(?i)\bSEARCH:\s*(.+)", stripped)
    if m_search:
        q = _parse_search_query("SEARCH: " + m_search.group(1))
        if q:
            return ("search", q)

    m_final = re.search(r"(?is)\bFINAL:\s*(.+)", stripped)
    if m_final:
        ans = m_final.group(1).strip()
        if ans:
            return ("final", ans)

    return ("unknown", None)


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


class AIProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        enable_thinking: bool = True,
        show_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        pass


class NvidiaLangChainProvider(AIProvider):
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
            return "💭 *Thinking:*\n" + "".join(reasoning_parts) + "\n\n" + "".join(content_parts)

        # Strip any <think>...</think> blocks the model may emit.
        cleaned = _strip_think_tags("".join(content_parts))
        if not cleaned:
            # Model put everything inside <think> (common on planner calls with short output).
            # Extract the content from within the think block as a fallback.
            m_think = re.search(r"<think>([\s\S]*?)</think>", "".join(content_parts), re.IGNORECASE)
            if m_think:
                cleaned = m_think.group(1).strip()
        if not cleaned and reasoning_parts:
            # Model generated only reasoning with no answer content (happens on follow-up planner
            # calls). Use the reasoning text so SEARCH:/FINAL: can still be parsed from it.
            cleaned = "".join(reasoning_parts).strip()
        return cleaned



class ExaSearchService:
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
    def _is_transient_error(e: Exception) -> bool:
        text = str(e).lower()
        return any(k in text for k in _TRANSIENT_ERROR_KEYWORDS)

    @staticmethod
    def _doc_to_snippet(doc) -> Optional[str]:
        title = (doc.metadata or {}).get("title", "").strip()
        url = (doc.metadata or {}).get("url", "").strip()
        highlights = (doc.metadata or {}).get("highlights", "")

        if isinstance(highlights, list):
            body = " ".join(str(h).strip() for h in highlights if str(h).strip())
        else:
            body = str(highlights).strip() if highlights else ""

        if not body:
            body = str(getattr(doc, "page_content", "") or "").strip()

        if len(body) < 15:
            return None

        if len(body) > EXA_MAX_SNIPPET_LEN:
            body = body[:EXA_MAX_SNIPPET_LEN]

        if title and url:
            return f"{title} ({url}): {body}"
        if title:
            return f"{title}: {body}"
        if url:
            return f"{url}: {body}"
        return body

    def search_sync(self, query: str) -> List[str]:
        if not self.api_keys:
            return []

        start_idx = self._next_index()
        last_err: Optional[Exception] = None

        for offset in range(len(self.api_keys)):
            key_idx = (start_idx + offset) % len(self.api_keys)
            api_key = self.api_keys[key_idx]
            try:
                retriever = ExaSearchRetriever(
                    exa_api_key=api_key,
                    k=EXA_MAX_RESULTS,
                    highlights=True,
                )

                docs = retriever.invoke(query)
                snippets: List[str] = []
                for doc in docs:
                    s = self._doc_to_snippet(doc)
                    if s:
                        snippets.append(s)

                logger.info(
                    "[Search] Exa key=%s query='%s' results=%s",
                    key_idx + 1,
                    query,
                    len(snippets),
                )
                return snippets
            except Exception as e:
                last_err = e
                logger.warning("[Search] Exa key=%s failed: %s", key_idx + 1, e)
                if not self._is_transient_error(e):
                    # Keep rotating: permanent failures are often key-specific
                    # (invalid/revoked key), and other keys may still succeed.
                    continue

        if last_err:
            logger.error("[Search] Exa failed for query '%s': %s", query, last_err)
        return []


provider = NvidiaLangChainProvider(NVIDIA_API_KEY or "")
search_service = ExaSearchService(EXA_API_KEYS)
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
    oldest = sorted(
        user_sessions.items(), key=lambda kv: kv[1].get("last_seen", now)
    )[:excess]
    for uid, _ in oldest:
        user_sessions.pop(uid, None)


def get_user_session(user_id: str) -> Dict:
    now = time.time()
    _prune_user_sessions(now)
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "thinking_enabled": False,
            "web_search": True,
            "last_seen": now,
            "cancel_event": asyncio.Event(),
        }
    else:
        user_sessions[user_id]["last_seen"] = now
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


async def _chat_with_retry(
    provider_obj,
    messages: list,
    enable_thinking: bool = True,
    show_thinking: bool = False,
    max_tokens: Optional[int] = None,
    cancel_event: Optional["asyncio.Event"] = None,
) -> str:
    for attempt in range(1, _CHAT_MAX_RETRIES + 2):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Cancelled by /restart")
        try:
            async with _get_semaphore():
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        provider_obj.chat,
                        messages=messages,
                        enable_thinking=enable_thinking,
                        show_thinking=show_thinking,
                        max_tokens=max_tokens or MAX_TOKENS,
                    ),
                    timeout=_CHAT_ATTEMPT_TIMEOUT,
                )
            return result or ""
        except asyncio.TimeoutError:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            e_for_delay = asyncio.TimeoutError()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            error_lower = str(e).lower()
            if not any(kw in error_lower for kw in _TRANSIENT_ERROR_KEYWORDS):
                raise
            e_for_delay = e

        delay = _CHAT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
        logger.warning(
            "[Bot] API transient error attempt %s/%s: %s. Retry in %ss",
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
    return ""


async def web_search(query: str) -> list:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(search_service.search_sync, query),
            timeout=EXA_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning("[Search] Exa search failed for '%s': %s", query, e)
        return []


async def _research_answer_loop(
    provider_obj,
    session_history: list,
    user_message: str,
    thinking_enabled: bool,
    cancel_event: Optional["asyncio.Event"] = None,
) -> str:
    async def _synthesize_from_traces(
        final_draft: Optional[str] = None,
    ) -> str:
        synth_snippets = [s for t in traces for s in t["snippets"]][:RESEARCH_MAX_SNIPPETS]
        synth_context = _build_search_context(
            query="iterative research",
            snippets=synth_snippets,
        )
        synth_msgs = session_history[:-1].copy()
        prompt_suffix = ""
        if final_draft:
            prompt_suffix = (
                "\n\nPlanner draft answer (improve and verify against snippets):\n"
                + final_draft
            )
        synth_msgs.append(
            {
                "role": "user",
                "content": user_message + "\n\n" + synth_context + prompt_suffix,
            }
        )
        # Remove any stale system message before inserting the search-enriched one.
        synth_msgs = [m for m in synth_msgs if m.get("role") != "system"]
        synth_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT_WITH_RESULTS})
        return await _chat_with_retry(
            provider_obj=provider_obj,
            messages=synth_msgs,
            enable_thinking=True,
            show_thinking=thinking_enabled,
            cancel_event=cancel_event,
        )

    traces: List[Dict[str, List[str]]] = []
    loop_prompt = _make_research_loop_prompt()

    for step in range(1, RESEARCH_MAX_STEPS + 1):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Cancelled by /restart")

        planner_messages = [{"role": "system", "content": loop_prompt}]
        planner_messages += [m for m in session_history[-MAX_HISTORY_MESSAGES:] if m.get("role") != "system"]
        planner_messages.append(
            {
                "role": "user",
                "content": (
                    f"Research step {step}/{RESEARCH_MAX_STEPS}.\n"
                    + _build_research_context(user_message, traces)
                ),
            }
        )

        decision = await _chat_with_retry(
            provider_obj=provider_obj,
            messages=planner_messages,
            enable_thinking=False,
            max_tokens=500,
            cancel_event=cancel_event,
        )
        action, payload = _parse_research_action(decision)
        logger.info("[Research] step=%s action=%s payload=%s", step, action, payload)

        if action == "final" and payload:
            # Enforce evidence-grounded output: require at least one search first.
            if not traces:
                fallback_q = _parse_search_query(user_message) or user_message[:80].strip()
                snippets = await web_search(fallback_q)
                traces.append(
                    {
                        "query": fallback_q,
                        "snippets": snippets or ["No useful results returned."],
                    }
                )
                continue
            return await _synthesize_from_traces(final_draft=payload)

        if action == "search" and payload:
            snippets = await web_search(payload)
            traces.append({"query": payload, "snippets": snippets or ["No useful results returned."]})
            continue

        # Malformed planner output.
        if not traces:
            # No evidence yet — do one fallback search so synthesis has something to work with.
            fallback_q = _parse_search_query(user_message) or user_message[:80].strip()
            logger.warning("[Research] step=%s unknown action, fallback search: '%s'", step, fallback_q)
            snippets = await web_search(fallback_q)
            traces.append({"query": fallback_q, "snippets": snippets or ["No useful results returned."]})
        else:
            # Already have evidence — skip this malformed step rather than repeating a bad query.
            logger.warning("[Research] step=%s unknown action, skipping (have %s trace(s))", step, len(traces))

    # Max loop steps reached: synthesize final answer from accumulated evidence.
    if traces:
        return await _synthesize_from_traces()

    # If no traces were generated, fallback to direct answer.
    return await _chat_with_retry(
        provider_obj=provider_obj,
        messages=session_history,
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
            return response[sep + 2:].strip()
    return response


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("\u26d4 Sorry, you're not authorized to use this bot.")
        return

    session = get_user_session(user_id)
    web_status = "ON \u2705" if session.get("web_search", True) else "OFF \U0001f515"
    think_status = "ON \u2705" if session.get("thinking_enabled", False) else "OFF \U0001f515"
    history_len = len([m for m in session.get("history", []) if m.get("role") != "system"])
    await update.message.reply_text(
        "\U0001f916 *NVIDIA + Exa Assistant*\n\n"
        f"\U0001f9e0 Model: `{MODEL_ID}`\n"
        f"\U0001f50e Web Search: {web_status}\n"
        f"\U0001f4ad Thinking: {think_status}\n"
        f"\U0001f4dc History: {history_len} messages\n\n"
        "*Commands:*\n"
        "`/status` \u2014 show current settings\n"
        "`/web on|off` \u2014 toggle web search\n"
        "`/thinking on|off` \u2014 toggle reasoning traces\n"
        "`/clear` \u2014 clear conversation history\n"
        "`/restart` \u2014 cancel a stuck request\n"
        "`/help` \u2014 full help",
        parse_mode="Markdown",
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    get_user_session(user_id)["history"] = []
    await update.message.reply_text("🗑️ Conversation history cleared!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    await update.message.reply_text(
        "💡 *How to use:*\n\n"
        "Send a message. The bot can run multiple search rounds before finalizing an answer.\n"
        "Web search is ON by default for a Perplexity-style, accuracy-first flow.\n\n"
        "*Commands:*\n"
        "• `/web` - show web search status\n"
        "• `/web on` / `/web off` - toggle Exa search\n"
        "• `/thinking on` / `/thinking off` - reasoning traces\n"
        "• `/clear` - clear conversation history\n"
        "• `/restart` - cancel pending request\n"
        "• `/help` - this message",
        parse_mode="Markdown",
    )


async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
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
        await update.message.reply_text("✅ *Thinking mode enabled.*", parse_mode="Markdown")
    elif arg == "off":
        session["thinking_enabled"] = False
        await update.message.reply_text("🔕 *Thinking mode disabled.*", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Use `/thinking on` or `/thinking off`", parse_mode="Markdown")


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return

    session = get_user_session(user_id)
    if not context.args:
        status = "ON" if session.get("web_search") else "OFF"
        await update.message.reply_text(
            f"🌐 *Web Search:* {status}\n"
            "🔎 *Engine:* Exa.ai (fixed)\n\n"
            "Use: `/web on` or `/web off`",
            parse_mode="Markdown",
        )
        return

    arg = context.args[0].lower()
    if arg == "on":
        session["web_search"] = True
        await update.message.reply_text("✅ Web search enabled (Exa.ai).")
    elif arg == "off":
        session["web_search"] = False
        await update.message.reply_text("🔕 Web search disabled.")
    else:
        await update.message.reply_text("❌ Use: `/web on|off`", parse_mode="Markdown")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return

    session = get_user_session(user_id)
    cancel_event: asyncio.Event = session["cancel_event"]
    cancel_event.set()

    history = session["history"]
    if history and history[-1].get("role") == "user":
        history.pop()

    await update.message.reply_text(
        "\U0001f6d1 *Restart requested.*\n\n"
        "Any pending AI request has been signalled to stop. You can send a new message now.",
        parse_mode="Markdown",
    )
    # Clear the event so the next incoming message is not immediately cancelled.
    cancel_event.clear()


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("\u26d4 Sorry, you're not authorized to use this bot.")
        return

    session = get_user_session(user_id)
    web = "ON \u2705" if session.get("web_search", True) else "OFF \U0001f515"
    thinking = "ON \u2705" if session.get("thinking_enabled", False) else "OFF \U0001f515"
    history_len = len([m for m in session.get("history", []) if m.get("role") != "system"])
    await update.message.reply_text(
        "\U0001f4ca *Current Settings*\n\n"
        f"\U0001f9e0 *Model:* `{MODEL_ID}`\n"
        f"\U0001f4ad *Thinking:* {thinking}\n"
        f"\U0001f310 *Web Search:* {web}\n"
        f"\U0001f4dc *History:* {history_len} messages\n"
        f"\U0001f511 *Exa keys:* {len(EXA_API_KEYS)}",
        parse_mode="Markdown",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
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

    try:
        if cancel_event.is_set():
            return

        # Keep the "typing…" indicator alive for the full duration of the request.
        typing_task = asyncio.create_task(_keep_typing(update.message.chat))

        thinking_enabled = session.get("thinking_enabled", False)
        web_on = session.get("web_search", True)

        session["history"].append({"role": "user", "content": user_message})

        skip = not web_on or _skip_search_decision(user_message)

        if skip:
            logger.info("[Bot] user=%s direct path (skip=%s)", user_id, skip)
            bot_response = await _chat_with_retry(
                provider_obj=provider,
                messages=session["history"],
                enable_thinking=True,
                show_thinking=thinking_enabled,
                cancel_event=cancel_event,
            )
        else:
            logger.info("[Bot] user=%s iterative research loop enabled", user_id)
            try:
                await update.message.reply_text("\U0001f50d Searching the web\u2026")
            except Exception:
                pass
            bot_response = await _research_answer_loop(
                provider_obj=provider,
                session_history=session["history"],
                user_message=user_message,
                thinking_enabled=thinking_enabled,
                cancel_event=cancel_event,
            )

        if not bot_response.strip():
            bot_response = "\u26a0\ufe0f The AI returned an empty response. Please try again."

        # Store a clean response (no thinking-trace header) to keep context window tidy.
        session["history"].append({"role": "assistant", "content": _clean_for_history(bot_response)})
        assistant_appended = True
        _trim_history(session["history"])

        if len(bot_response) <= MAX_MESSAGE_LENGTH:
            await reply_text_safe(update.message, bot_response)
        else:
            header_reserve = 25
            chunk_limit = MAX_MESSAGE_LENGTH - header_reserve
            chunks: List[str] = []
            current_chunk = ""
            for line in bot_response.split("\n"):
                if len(line) > chunk_limit:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    # Split long lines at word boundaries to avoid breaking mid-word.
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
                    current_chunk = ((current_chunk + "\n" + line) if current_chunk else line)
            if current_chunk:
                chunks.append(current_chunk)

            for i, chunk in enumerate(chunks, 1):
                header = f"\U0001f4c4 Part {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
                await reply_text_safe(update.message, header + chunk)

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
        try:
            await update.message.reply_text(
                "\u274c Something went wrong processing your request.\n\n"
                "Try:\n\u2022 `/clear` to reset conversation\n\u2022 `/restart` to cancel stuck calls",
                parse_mode="Markdown",
            )
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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 NVIDIA + Exa Telegram bot started")
    logger.info("🧠 Model: %s", MODEL_ID)
    logger.info("🔎 Exa keys loaded: %s", len(EXA_API_KEYS))
    logger.info("🌐 Web search default: ON")
    logger.info(
        "⚙️ Research controls: steps=%s snippets=%s llm_concurrency=%s",
        RESEARCH_MAX_STEPS,
        RESEARCH_MAX_SNIPPETS,
        LLM_MAX_CONCURRENCY,
    )
    logger.info(
        "🧹 Session controls: ttl_seconds=%s max_sessions=%s",
        SESSION_TTL_SECONDS,
        MAX_USER_SESSIONS,
    )
    if ALLOWED_USER_IDS:
        logger.info("🔐 Access mode: allowlist enabled (%s Telegram UID(s))", len(ALLOWED_USER_IDS))
    else:
        logger.info("🔓 Access mode: open (no TELEGRAM_ALLOWED_UIDS set)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()