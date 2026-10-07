"""
G.I.D.E.O.N IP-API Geolocation Tool

Retrieves IP geolocation diagnostic data using ip-api.com.
Free endpoint, no API key required.
"""

from __future__ import annotations

import json
import urllib.request

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.geolocation")


@tool(
    "get_ip_info",
    "Lookup IP address geolocation, ISP, and network details. Call when user asks for their public IP, location info, or IP lookup.",
    Param(
        "ip_address",
        "Optional IP address to look up. Omit for current device public IP.",
        aliases=("ip",),
        default=None,
    ),
    triggers={"ip", "location", "where", "geolocation", "isp", "timezone"},
)
def get_ip_info(ip_address: str | None = None) -> str:
    """Retrieve IP address geolocation details (City, Region, Country, ISP, Lat/Lon)."""
    target = (ip_address or "").strip()
    url = f"http://ip-api.com/json/{target}" if target else "http://ip-api.com/json/"

    req = urllib.request.Request(url, headers={"User-Agent": "GIDEON-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") != "success":
            msg = data.get("message", "Unknown error")
            return f"IP lookup failed: {msg}"

        query_ip = data.get("query", target or "Current IP")
        city = data.get("city", "Unknown")
        region = data.get("regionName", data.get("region", ""))
        country = data.get("country", "Unknown")
        lat = data.get("lat")
        lon = data.get("lon")
        isp = data.get("isp", "Unknown")
        timezone = data.get("timezone", "Unknown")

        return (
            f"IP Geolocation for {query_ip}:\n"
            f"- Location: {city}, {region}, {country}\n"
            f"- Coordinates: Latitude {lat}, Longitude {lon}\n"
            f"- Timezone: {timezone}\n"
            f"- ISP: {isp}"
        )

    except Exception as err:
        log.error("IP-API request failed: %s", err)
        return f"Failed to retrieve IP geolocation info: {err}"
