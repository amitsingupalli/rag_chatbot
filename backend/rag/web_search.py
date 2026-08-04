"""Internet search for current affairs and real-time information."""

from __future__ import annotations

import logging
import re
import urllib.parse
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
    "web_search",
    "web search",
    "do search",
    "search online",
    "agentic ai",
    "agentic",
    "ai",
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
    "web_search",
    "web search",
    "do the web",
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


def _clean_search_query(message: str) -> str:
    cleaned = message.strip()
    patterns = [
        r"(?i)^(?:do\s+(?:a\s+|the\s+)?)?(?:web_search|web\s+search|internet\s+search|google\s+search)\s*(?:and\s*)?(?:tell\s+me\s+about\s*)?",
        r"(?i)^(?:search\s+(?:the\s+web\s+for|for|online\s+for)|browse\s+for|look\s+up|find\s+online|check\s+online)\s*",
        r"(?i)^(?:tell\s+me\s+about|what\s+is\s+meant\s+by|what\s+is|who\s+is|explain)\s*",
        r"(?i)^(?:the|a|an)\s+",
    ]
    for p in patterns:
        cleaned = re.sub(p, "", cleaned).strip()
    return cleaned if len(cleaned) > 2 else message


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
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            text = _strip_html(response.text)
            return text[: settings.web_page_max_chars]
    except Exception as exc:
        logger.debug("Could not fetch %s: %s", url, exc)
        return ""


def _search_duckduckgo_html(query: str, max_results: int = 5) -> list[WebSearchResult]:
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    results: list[WebSearchResult] = []
    try:
        resp = httpx.post(url, data={"q": query}, headers=headers, follow_redirects=True, timeout=10.0)
        if resp.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.find_all("div", class_="result"):
                title_elem = result.find("a", class_="result__a")
                snippet_elem = result.find("a", class_="result__snippet")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get("href", "")
                    if "uddg=" in link:
                        link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    if title and link:
                        results.append(WebSearchResult(title=title, url=link, snippet=snippet))
                    if len(results) >= max_results:
                        break
    except Exception as exc:
        logger.warning("DuckDuckGo HTML POST search failed: %s", exc)
    return results


def _search_duckduckgo(query: str, max_results: int) -> list[WebSearchResult]:
    results = _search_duckduckgo_html(query, max_results=max_results)
    if not results:
        try:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                from ddgs import DDGS

            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    results.append(
                        WebSearchResult(
                            title=item.get("title", ""),
                            url=item.get("href", item.get("link", "")),
                            snippet=item.get("body", item.get("snippet", "")),
                        )
                    )
        except Exception as exc:
            logger.warning("DDGS API search fallback failed: %s", exc)

    return results


def search_web(query: str, max_results: int | None = None) -> tuple[str, list[str]]:
    """Search the web and return formatted context plus source URLs."""
    if not settings.web_search_enabled:
        return "", []

    clean_q = _clean_search_query(query)
    max_results = max_results or settings.web_search_max_results

    try:
        hits = _search_duckduckgo(clean_q, max_results=max_results)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return f"[Web search unavailable: {exc}]", []

    if not hits:
        return "", []

    if settings.web_fetch_pages:
        for hit in hits[: settings.web_fetch_top_n]:
            if hit.url and hit.url.startswith("http"):
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

    context = f"Live web search results for '{clean_q}':\n\n" + "\n\n---\n\n".join(blocks)
    return context, list(dict.fromkeys(s for s in sources if s))
