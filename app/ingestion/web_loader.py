import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

from app.ingestion.pdf_loader import normalize_text

DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_HTML_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; DocumentIntelligenceRAG/0.1; +https://github.com/mhadeli/document-rag)"
)


class WebExtractionError(Exception):
    """Base exception for website extraction errors."""


class InvalidURLError(WebExtractionError):
    """Raised when a URL has invalid syntax or an unsupported scheme."""


class SSRFBlockedError(WebExtractionError):
    """Raised when a URL points to a private, loopback, or internal network address."""


class WebFetchError(WebExtractionError):
    """Raised when fetching the webpage fails or exceeds size/time limits."""


class EmptyWebContentError(WebExtractionError):
    """Raised when no readable article content can be extracted from the webpage."""


@dataclass(slots=True)
class ExtractedWebsite:
    source_url: str
    title: str
    content: str
    source_type: str = "website"


def _check_ip_is_public(ip_str: str, hostname: str) -> None:
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        raise InvalidURLError(f"Invalid IP address {ip_str!r} for host {hostname!r}.") from exc

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        raise SSRFBlockedError(
            f"Access to private, loopback, or internal network address is blocked: {hostname}"
        )


def validate_url_and_check_ssrf(url: str, *, resolve_dns: bool = True) -> str:
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        raise InvalidURLError("URL cannot be empty.")

    parsed = urlparse(cleaned_url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise InvalidURLError("URL must start with http:// or https://.")

    hostname = parsed.hostname
    if not hostname:
        raise InvalidURLError("URL must include a valid hostname.")

    lower_host = hostname.lower()
    if lower_host in ("localhost", "localhost.localdomain") or lower_host.endswith(
        (".local", ".internal")
    ):
        raise SSRFBlockedError(f"Access to local/internal host is blocked: {hostname}")

    # Check if hostname is directly an IP literal
    try:
        ipaddress.ip_address(lower_host)
        _check_ip_is_public(lower_host, hostname)
        return cleaned_url
    except ValueError:
        pass

    if resolve_dns:
        try:
            addr_info = socket.getaddrinfo(hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise WebFetchError(f"Could not resolve hostname '{hostname}': {exc}") from exc

        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            _check_ip_is_public(ip_str, hostname)

    return cleaned_url


def _get_field(extracted: Any, key: str) -> str | None:
    if isinstance(extracted, dict):
        val = extracted.get(key)
    else:
        val = getattr(extracted, key, None)
    return str(val).strip() if val else None


def extract_website_from_html(html: str, source_url: str) -> ExtractedWebsite:
    if not html or not html.strip():
        raise EmptyWebContentError(f"Empty HTML response from '{source_url}'.")

    extracted = trafilatura.bare_extraction(
        html,
        url=source_url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        with_metadata=True,
    )

    raw_text: str | None = None
    title: str | None = None
    canonical_url: str | None = None

    if extracted is not None:
        raw_text = _get_field(extracted, "text")
        title = _get_field(extracted, "title")
        canonical_url = _get_field(extracted, "url")

    if not title:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            title = normalize_text(title_match.group(1))

    if not raw_text:
        raw_text = trafilatura.extract(
            html,
            url=source_url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )

    cleaned_text = normalize_text(raw_text or "")
    if not cleaned_text:
        raise EmptyWebContentError(f"No readable content could be extracted from '{source_url}'.")

    resolved_url = canonical_url or source_url
    parsed_fallback = urlparse(source_url).netloc or source_url
    resolved_title = title or parsed_fallback

    return ExtractedWebsite(
        source_url=resolved_url,
        title=resolved_title,
        content=cleaned_text,
    )


def extract_website(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_HTML_BYTES,
    transport: httpx.BaseTransport | None = None,
    resolve_dns: bool = True,
) -> ExtractedWebsite:
    validated_url = validate_url_and_check_ssrf(url, resolve_dns=resolve_dns)

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=5,
            headers=headers,
            transport=transport,
        ) as client:
            with client.stream("GET", validated_url) as response:
                validate_url_and_check_ssrf(str(response.url), resolve_dns=resolve_dns)

                if response.status_code >= 400:
                    raise WebFetchError(
                        f"Failed to fetch '{validated_url}': HTTP {response.status_code}"
                    )

                chunks: list[bytes] = []
                total_bytes = 0
                for chunk in response.iter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise WebFetchError(
                            f"Webpage '{validated_url}' exceeds maximum size of {max_bytes} bytes."
                        )
                    chunks.append(chunk)

                encoding = response.encoding or "utf-8"
                html = b"".join(chunks).decode(encoding, errors="replace")
                final_url = str(response.url)
    except WebExtractionError:
        raise
    except httpx.HTTPError as exc:
        raise WebFetchError(f"HTTP error while fetching '{validated_url}': {exc}") from exc

    return extract_website_from_html(html, source_url=final_url)
