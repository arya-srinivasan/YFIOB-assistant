import os
import hashlib
import requests
from datetime import datetime
from typing import Optional
from .schema import CareerEvent
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY      = os.getenv("SERPAPI_KEY")
SERPAPI_ENDPOINT = "https://serpapi.com/search"

CAREER_KEYWORDS = [
    "career fair", "job fair", "hiring event",
    "networking event", "career expo", "recruitment event",
]


def _make_id(title: str, location: str) -> str:
    return hashlib.md5(f"{title}{location}".lower().encode()).hexdigest()


def google_search_career_events(
    location: str,
    industry: Optional[str] = None,
    num_results: int = 10,
) -> list[CareerEvent]:
    """
    Search Google via SerpAPI for career events in a given location.
    """

    all_events = []
    seen_ids = set()
    for keyword in CAREER_KEYWORDS:
        query = f"({keyword})"
        if industry:
            query += f" {industry}"

    params = {
        "api_key": SERPAPI_KEY,
        "engine":  "google",
        "q":       query,
        "num":     5,
        "tbs":     "qdr:m3",
    }

    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("organic_results", [])

        for item in items:
            title = item.get("title", "").strip()
            if not title or len(title) < 10 or title.lower() in ("events", "home", "jobs"):
                continue
            event_id = _make_id(title, location)
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)
            all_events.append(CareerEvent(
                id=_make_id(title, location),
                title=title,
                location=location,
                industry=industry or "General",
                description=item.get("snippet", "").strip(),
                registration_url=item.get("link", "").strip(),
                source="google",
                scraped_at=datetime.now().isoformat(),
            ))
    except Exception as e:
        print(f"[scraper] Failed keyword '{keyword}': {e}")

    print(f"[scraper] Found {len(all_events)} unique events for {location}")
    return all_events