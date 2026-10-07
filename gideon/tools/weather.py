"""
G.I.D.E.O.N Open-Meteo Weather Tool

Retrieves live weather data and short-term forecasts using Open-Meteo's free REST API.
Does not require any API keys.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool
from gideon.tools.geolocation import get_ip_info

log = get_logger("gideon.tools.weather")

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode_location(location_name: str) -> tuple[float, float, str, str] | None:
    """Geocode a city or place name using Open-Meteo Geocoding API.
    Returns (lat, lon, display_name, country).
    """
    encoded = urllib.parse.quote(location_name.strip())
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"

    req = urllib.request.Request(url, headers={"User-Agent": "GIDEON-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results")
            if results and len(results) > 0:
                item = results[0]
                return (
                    float(item["latitude"]),
                    float(item["longitude"]),
                    item.get("name", location_name),
                    item.get("country", ""),
                )
    except Exception as err:
        log.warning("Geocoding failed for %r: %s", location_name, err)

    return None


@tool(
    "get_weather",
    "Fetch live weather conditions and daily temperature forecast. Call when user asks about weather, temperature, rain, or forecast for a city or local area.",
    Param(
        "location",
        "City or location name (e.g. 'Tokyo', 'London', 'New York'). Omit for current location.",
        aliases=("city", "place"),
        default=None,
    ),
    triggers={
        "weather", "forecast", "rain", "raining", "snow", "sunny", "cloudy",
        "humidity", "hot", "cold", "windy", "degrees", "celsius", "fahrenheit",
        "climate",
    },
)
def get_weather(location: str | None = None) -> str:
    """Fetch current weather and forecast for a given location or current IP location."""
    lat: float
    lon: float
    city_name: str = ""
    country_name: str = ""

    if location and location.strip():
        geocoded = _geocode_location(location)
        if geocoded:
            lat, lon, city_name, country_name = geocoded
        else:
            return f"Unable to locate or geocode '{location}'. Please check the city name."
    else:
        # Fallback to IP geolocation
        ip_info_str = get_ip_info()
        # Parse city name from geolocation string line 2 if present
        # Format: "- Location: City, Region, Country"
        extracted_city = ""
        for line in ip_info_str.splitlines():
            if line.startswith("- Location:"):
                parts = line.split(":", 1)[1].split(",")
                if parts:
                    extracted_city = parts[0].strip()
                break

        if extracted_city:
            geocoded = _geocode_location(extracted_city)
            if geocoded:
                lat, lon, city_name, country_name = geocoded
            else:
                return f"Could not determine weather for automatically resolved location: {extracted_city}."
        else:
            return f"Location not specified and automatic IP location lookup failed. Details: {ip_info_str}"

    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,"
        f"weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,weather_code"
        f"&timezone=auto"
    )

    req = urllib.request.Request(forecast_url, headers={"User-Agent": "GIDEON-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        wmo_code = current.get("weather_code", 0)
        condition = WMO_WEATHER_CODES.get(int(wmo_code), "Unknown")

        daily = data.get("daily", {})
        high = daily.get("temperature_2m_max", [None])[0]
        low = daily.get("temperature_2m_min", [None])[0]

        loc_label = f"{city_name}, {country_name}".strip(", ") if city_name else f"Lat {lat}, Lon {lon}"

        res_lines = [
            f"Weather for {loc_label}:",
            f"- Condition: {condition}",
            f"- Temperature: {temp}°C",
            f"- Humidity: {humidity}%",
            f"- Wind Speed: {wind} km/h",
        ]
        if high is not None and low is not None:
            res_lines.append(f"- Today's Forecast: High {high}°C / Low {low}°C")

        return "\n".join(res_lines)

    except Exception as err:
        log.error("Failed to fetch Open-Meteo forecast: %s", err)
        return f"Error retrieving weather data for {city_name or location}: {err}"
