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
    keyword_query = " OR ".join(f'"{kw}"' for kw in CAREER_KEYWORDS)
    query = f"({keyword_query}) {industry or ''} {location}".strip()

    params = {
        "api_key": SERPAPI_KEY,
        "engine":  "google",
        "q":       query,
        "num":     num_results,
        "tbs":     "qdr:m3",
    }

    response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=10)
    response.raise_for_status()
    items = response.json().get("organic_results", [])

    events = []
    for item in items:
        title = item.get("title", "").strip()
        events.append(CareerEvent(
            id=_make_id(title, location),
            title=title,
            location=location,
            industry=industry,
            description=item.get("snippet", "").strip(),
            registration_url=item.get("link", "").strip(),
            source="google",
            scraped_at=datetime.now().isoformat(),
        ))

    return events