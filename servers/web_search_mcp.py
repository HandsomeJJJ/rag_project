from __future__ import annotations

import re
from urllib.parse import urljoin

import requests
from mcp.server.fastmcp import FastMCP

#用于模拟浏览器的 User-Agent，避免被搜索引擎屏蔽
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)

mcp = FastMCP("web-search", json_response=True)

# DuckDuckGo 的 HTML 搜索接口，返回结果较为简洁，适合快速抓取和解析
def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    resp.raise_for_status()
    html = resp.text

    items: list[dict[str, str]] = []
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html,
        flags=re.S,
    )

    for href, title_html, snippet_html in blocks[:max_results]:
        title = re.sub(r"<.*?>", "", title_html)
        snippet = re.sub(r"<.*?>", "", snippet_html)
        items.append(
            {
                "title": re.sub(r"\s+", " ", title).strip(),
                "url": urljoin("https://duckduckgo.com", href),
                "snippet": re.sub(r"\s+", " ", snippet).strip(),
            }
        )

    return items

# 定义 MCP 工具，暴露给外部调用
@mcp.tool()
def web_search(query: str, max_results: int = 5) -> dict[str, object]:
    """Search the web and return a compact result list."""
    max_results = max(1, min(int(max_results), 10))
    results = _search_duckduckgo(query, max_results)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run()

