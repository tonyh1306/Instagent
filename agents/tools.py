"""Concrete tool implementations shared across agent subclasses.

Each function here is paired with an OpenAI-style tool schema (TOOL_SCHEMAS) so
agent subclasses can pick the schema + executor for whatever tools they're scoped to.
"""

import subprocess
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WORKSPACE_DIR = (Path(__file__).resolve().parent.parent / "workspace").resolve()
WORKSPACE_DIR.mkdir(exist_ok=True)

HTTP_TIMEOUT_S = 15
MAX_FETCH_CHARS = 4000
MAX_SEARCH_RESULTS = 5


def _resolve_in_workspace(relative_path: str) -> Path:
    # Agents are prone to echoing "workspace/foo.py" when the workspace itself is
    # what they've been told to write into; strip a redundant leading segment.
    parts = Path(relative_path).parts
    if parts and parts[0] == "workspace":
        relative_path = str(Path(*parts[1:])) if len(parts) > 1 else "."
    candidate = (WORKSPACE_DIR / relative_path).resolve()
    if WORKSPACE_DIR not in candidate.parents and candidate != WORKSPACE_DIR:
        raise ValueError(f"path {relative_path!r} escapes the workspace sandbox")
    return candidate


def file_read(path: str) -> dict:
    """Read a file's content from the sandboxed workspace directory. Read-only."""
    target = _resolve_in_workspace(path)
    if not target.exists():
        return {"error": f"{path!r} does not exist in the workspace"}
    return {"path": path, "content": target.read_text()[:MAX_FETCH_CHARS]}


def file_write(path: str, content: str) -> dict:
    """Write `content` to `path`, resolved relative to the sandboxed workspace directory."""
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"written": str(target.relative_to(WORKSPACE_DIR)), "bytes": len(content)}


def run_tests(path: str = ".") -> dict:
    """Run pytest against `path` (relative to the workspace) and return the result."""
    target = _resolve_in_workspace(path)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-q"],
        cwd=WORKSPACE_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-3000:],
        "stderr": proc.stderr[-2000:],
    }


def lint(path: str = ".") -> dict:
    """Run ruff check against `path` (relative to the workspace) and return the result."""
    target = _resolve_in_workspace(path)
    proc = subprocess.run(
        ["ruff", "check", str(target)],
        cwd=WORKSPACE_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-3000:],
        "stderr": proc.stderr[-1000:],
    }


def web_search(query: str) -> dict:
    """Search the web (DuckDuckGo HTML front-end, no API key required) and return top results."""
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; qwen-agents/0.1)"},
        timeout=HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for link in soup.select("a.result__a")[:MAX_SEARCH_RESULTS]:
        results.append({"title": link.get_text(strip=True), "url": link.get("href")})
    return {"query": query, "results": results}


def doc_fetch(url: str) -> dict:
    """Fetch a URL and return its extracted text content, truncated."""
    resp = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; qwen-agents/0.1)"}, timeout=HTTP_TIMEOUT_S
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = " ".join(soup.get_text(separator=" ").split())
    return {"url": url, "text": text[:MAX_FETCH_CHARS]}


def doc_format(content: str, style: str = "markdown") -> dict:
    """Normalize whitespace in a document body (single trailing newline, no trailing spaces)."""
    lines = [line.rstrip() for line in content.splitlines()]
    normalized = "\n".join(lines).strip() + "\n"
    return {"style": style, "content": normalized}


def schema_validate(artifact: dict, required_keys: list[str]) -> dict:
    """Check that `artifact` contains all of `required_keys`. Read-only, used by the critic."""
    missing = [k for k in required_keys if k not in artifact]
    return {"valid": not missing, "missing": missing}


TOOL_SCHEMAS = {
    "file_read": {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a file's content from the sandboxed workspace directory. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                },
                "required": ["path"],
            },
        },
    },
    "file_write": {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file in the sandboxed workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    "run_tests": {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest against a path in the workspace and return pass/fail output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root (default '.')."},
                },
                "required": [],
            },
        },
    },
    "lint": {
        "type": "function",
        "function": {
            "name": "lint",
            "description": "Run ruff check against a path in the workspace and return findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root (default '.')."},
                },
                "required": [],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a short list of title/url results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "doc_fetch": {
        "type": "function",
        "function": {
            "name": "doc_fetch",
            "description": "Fetch a URL and return its extracted text content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    "doc_format": {
        "type": "function",
        "function": {
            "name": "doc_format",
            "description": "Normalize whitespace/formatting of a document body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "style": {"type": "string", "description": "e.g. 'markdown' (default)"},
                },
                "required": ["content"],
            },
        },
    },
    "schema_validate": {
        "type": "function",
        "function": {
            "name": "schema_validate",
            "description": "Check that a JSON artifact contains all required keys. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact": {"type": "object", "description": "The artifact to validate."},
                    "required_keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["artifact", "required_keys"],
            },
        },
    },
}
