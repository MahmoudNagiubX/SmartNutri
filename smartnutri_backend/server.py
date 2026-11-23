"""Explicit local/LAN launcher for the SmartNutri API."""

from __future__ import annotations

import ipaddress
import os


_RFC1918_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def resolve_bind_host(value: str | None) -> str:
    """Allow loopback or an explicit RFC1918 IPv4 address only."""
    candidate = "127.0.0.1" if value is None else value.strip()
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise ValueError("SMARTNUTRI_API_HOST must be an IPv4 address") from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ValueError("SMARTNUTRI_API_HOST must be an IPv4 address")
    if address.is_loopback:
        return str(address)
    if any(address in network for network in _RFC1918_NETWORKS):
        return str(address)
    raise ValueError("SMARTNUTRI_API_HOST must be loopback or an explicit RFC1918 LAN address")


def resolve_port(value: str | None) -> int:
    """Parse a valid TCP port from SMARTNUTRI_API_PORT."""
    candidate = "8000" if value is None else value.strip()
    try:
        port = int(candidate)
    except ValueError as exc:
        raise ValueError("SMARTNUTRI_API_PORT must be an integer in 1..65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("SMARTNUTRI_API_PORT must be an integer in 1..65535")
    return port


def main() -> None:
    host = resolve_bind_host(os.environ.get("SMARTNUTRI_API_HOST"))
    port = resolve_port(os.environ.get("SMARTNUTRI_API_PORT"))

    import uvicorn

    uvicorn.run("smartnutri_backend.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
