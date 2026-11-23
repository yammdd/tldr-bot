"""SSRF protection and URL normalization for user-supplied OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse, urlunparse


class EndpointSecurityError(Exception):
    """Raised when a custom endpoint URL fails security or syntax validation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _is_blocked_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def normalize_base_url(raw_url: str) -> str:
    """Trim whitespace and strip accidental '/chat/completions' suffix from an OpenAI-style base URL."""
    cleaned = (raw_url or "").strip()
    parsed = urlparse(cleaned)
    path = parsed.path or ""
    for suffix in ("/chat/completions", "/chat/completions/"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    # Keep a single clean path without duplicate trailing slashes
    path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc, path, "", "", ""))


def validate_endpoint_url(
    raw_url: str,
    *,
    allow_local: bool = False,
    resolve_dns: bool = True,
) -> str:
    """Validate and normalize an LLM API base URL, blocking SSRF targets by default."""
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise EndpointSecurityError("err_ssrf_invalid", "Empty endpoint URL.")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if not allow_local and scheme != "https":
        raise EndpointSecurityError("err_ssrf_scheme", "Only https:// endpoints are allowed.")
    if allow_local and scheme not in {"https", "http"}:
        raise EndpointSecurityError("err_ssrf_scheme", "Endpoint must use https:// or http://.")

    if parsed.username or parsed.password:
        raise EndpointSecurityError("err_ssrf_invalid", "Embedded credentials in URL are forbidden.")
    if parsed.query or parsed.fragment:
        raise EndpointSecurityError("err_ssrf_invalid", "Query parameters or fragments are not allowed in base URL.")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise EndpointSecurityError("err_ssrf_invalid", "Endpoint hostname is missing.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise EndpointSecurityError("err_ssrf_invalid", "Invalid port in endpoint URL.") from exc

    if not allow_local:
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
            raise EndpointSecurityError("err_ssrf_private", "Localhost endpoints are blocked.")

        # Direct IP literal check
        ip_obj = None
        try:
            ip_obj = ipaddress.ip_address(hostname)
        except ValueError:
            ip_obj = None

        if ip_obj is not None:
            if _is_blocked_ip(hostname):
                raise EndpointSecurityError("err_ssrf_private", "Private or loopback IP blocked.")
        elif resolve_dns:
            # Hostname is a domain name; resolve via DNS if requested
            try:
                addr_infos = socket.getaddrinfo(
                    hostname, port or 443, type=socket.SOCK_STREAM
                )
            except socket.gaierror as exc:
                raise EndpointSecurityError(
                    "err_ssrf_dns", f"Failed to resolve host {hostname}."
                ) from exc

            if not addr_infos:
                raise EndpointSecurityError(
                    "err_ssrf_dns", f"No DNS records returned for {hostname}."
                )

            for info in addr_infos:
                sockaddr = info[4]
                resolved_ip = sockaddr[0]
                if _is_blocked_ip(resolved_ip):
                    raise EndpointSecurityError(
                        "err_ssrf_private",
                        "Resolved IP belongs to a private or reserved network.",
                    )

    return normalize_base_url(cleaned)


async def validate_endpoint_url_async(
    raw_url: str,
    *,
    allow_local: bool = False,
    resolve_dns: bool = True,
) -> str:
    """Async wrapper around validate_endpoint_url so DNS resolution never blocks the event loop."""
    return await asyncio.to_thread(
        validate_endpoint_url,
        raw_url,
        allow_local=allow_local,
        resolve_dns=resolve_dns,
    )
