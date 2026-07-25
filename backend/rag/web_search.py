"""Internet search for current affairs and real-time information."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

CURRENT_AFFAIRS_KEYWORDS = (
    "today",
    "latest",
    "recent",
    "current",
    "news",
    "right now",
    "this week",
    "this month",
    "this year",
    "2024",
    "2025",
    "2026",
    "happening",
    "update",
    "live score",
    "weather",
    "stock price",
    "election",
    "who won",
    "who is the",
    "current president",
    "current pm",
    "prime minister",
    "breaking",
    "headline",
    "search the web",
    "browse the web",
    "look up online",
    "on the internet",
    "google",
    "current affairs",
    "as of now",
)

EXPLICIT_SEARCH_PHRASES = (
    "search for",
    "search the web",
    "browse for",
    "look up",
    "find online",
    "check online",
    "what is happening",
    "what happened",
)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    body: str = ""


def should_search_web(message: str, force: bool | None = None) -> bool:
    if force is True:
        return True
    if force is False:
        return False

    lower = message.lower()
    if any(phrase in lower for phrase in EXPLICIT_SEARCH_PHRASES):
        return True
    if any(kw in lower for kw in CURRENT_AFFAIRS_KEYWORDS):
        return True

    if re.search(r"\b(20\d{2})\b", lower):
        return True

    return False


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<.*?>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_page_text(url: str, timeout: float = 8.0) -> str:
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAGChatbot/1.0)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            text = _strip_html(response.text)
            return text[: settings.web_page_max_chars]
    except Exception as exc:
        logger.debug("Could not fetch %s: %s", url, exc)
        return ""


def _search_duckduckgo(query: str, max_results: int) -> list[WebSearchResult]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "duckduckgo-search is not installed. Run: pip install duckduckgo-search"
        ) from exc

    results: list[WebSearchResult] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", item.get("snippet", "")),
                )
            )
    return results


def search_web(query: str, max_results: int | None = None) -> tuple[str, list[str]]:
    """Search the web and return formatted context plus source URLs."""
    if not settings.web_search_enabled:
        return "", []

    max_results = max_results or settings.web_search_max_results
    try:
        hits = _search_duckduckgo(query, max_results=max_results)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return f"[Web search unavailable: {exc}]", []

    if not hits:
        return "", []

    if settings.web_fetch_pages:
        for hit in hits[: settings.web_fetch_top_n]:
            if hit.url:
                hit.body = _fetch_page_text(hit.url)

    blocks: list[str] = []
    sources: list[str] = []
    for i, hit in enumerate(hits, start=1):
        host = urlparse(hit.url).netloc or hit.url
        sources.append(hit.url or hit.title or host)
        block = f"Result {i}: {hit.title}\nURL: {hit.url}\nSummary: {hit.snippet}"
        if hit.body:
            block += f"\nPage excerpt: {hit.body[:1500]}"
        blocks.append(block)

    context = "Live web search results:\n\n" + "\n\n---\n\n".join(blocks)
    return context, list(dict.fromkeys(s for s in sources if s))
