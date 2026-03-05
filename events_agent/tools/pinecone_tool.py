import os
from datetime import datetime
from typing import Optional
from pinecone import Pinecone
from pydantic import BaseModel
from .schema import CareerEvent
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NAMESPACE           = "career-events"

_index = None

def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index_for_model(
                name=PINECONE_INDEX_NAME,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "text"},
                }
            )
        _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_embed_text(event: CareerEvent) -> str:
    """Combine event fields into a single string for Pinecone to embed."""
    companies = ", ".join(event.companies_attending) if event.companies_attending else ""
    majors    = ", ".join(event.target_majors) if event.target_majors else ""
    return "\n".join(filter(None, [
        f"Title: {event.title}",
        f"Type: {event.event_type or ''}",
        f"Industry: {event.industry or 'General'}",
        f"Location: {event.location}",
        f"Virtual: {'Yes' if event.is_virtual else 'No'}",
        f"Date: {event.date or 'Unknown'}",
        f"Experience Level: {event.experience_level or ''}",
        f"Target Majors: {majors}",
        f"Companies Attending: {companies}",
        f"Organizer: {event.organizer or ''}",
        f"Description: {event.description or ''}",
    ]))


def _date_score(date_str: str) -> float:
    """Score event by date proximity. Upcoming events score highest."""
    if not date_str:
        return 0.3
    try:
        days_diff = (datetime.fromisoformat(date_str) - datetime.now()).days
        if days_diff < 0:      return max(0.0, 0.4 + days_diff * 0.01)
        elif days_diff <= 7:   return 1.0
        elif days_diff <= 30:  return 0.85
        elif days_diff <= 90:  return 0.65
        else:                  return 0.4
    except (ValueError, TypeError):
        return 0.3


# ── Output Schema ──────────────────────────────────────────────────────────

class RankedEvent(BaseModel):
    title: str
    date: str
    location: str
    venue: Optional[str] = None
    is_virtual: bool = False
    industry: str
    event_type: Optional[str] = None
    experience_level: Optional[str] = None
    companies_attending: Optional[list[str]] = None
    target_majors: Optional[list[str]] = None
    organizer: Optional[str] = None
    description: str
    registration_url: str
    source: str
    score: float
    similarity_score: float
    date_score: float


def ingest_events_to_pinecone(events: list[CareerEvent]) -> dict:
    """
    Upsert career events into Pinecone. Pinecone automatically embeds
    the text field using llama-text-embed-v2.

    Args:
        events: List of CareerEvent objects to ingest.

    Returns:
        Summary dict with upserted and failed counts.
    """
    index   = _get_index()
    records = []
    failed  = 0

    for event in events:
        try:
            records.append({
                "id":               event.id,
                "text":             _build_embed_text(event),  # Pinecone embeds this
                "title":            event.title,
                "date":             event.date or "",
                "end_date":         event.end_date or "",
                "location":         event.location,
                "venue":            event.venue or "",
                "is_virtual":       event.is_virtual,
                "industry":         event.industry or "General",
                "event_type":       event.event_type or "",
                "experience_level": event.experience_level or "",
                "target_majors":    event.target_majors or [],
                "companies_attending": event.companies_attending or [],
                "organizer":        event.organizer or "",
                "organizer_type":   event.organizer_type or "",
                "description":      event.description or "",
                "registration_url": event.registration_url or "",
                "source":           event.source,
                "scraped_at":       event.scraped_at,
            })
        except Exception as e:
            print(f"[ingest] Failed '{event.title}': {e}")
            failed += 1

    if records:
        index.upsert_records(namespace=NAMESPACE, records=records)

    upserted = len(events) - failed
    return {"upserted": upserted, "failed": failed, "total": len(events)}


def retrieve_career_events(
    query: str,
    industry: Optional[str] = None,
    location: Optional[str] = None,
    top_k: int = 5,
    date_weight: float = 0.4,
    similarity_weight: float = 0.6,
) -> list[RankedEvent]:
    """
    Retrieve and rank career events from Pinecone based on a student profile query.

    Ranking combines semantic similarity (default 60%) and date proximity (default 40%).

    Args:
        query:             Natural language student profile e.g.
                           'Junior CS student interested in AI in Austin, TX'.
        industry:          Filter by industry e.g. 'Tech'.
        location:          Filter by city e.g. 'Austin, TX'.
        event_type:        Filter by type e.g. 'career_fair', 'networking'.
        experience_level:  Filter by level e.g. 'internship', 'entry_level'.
        include_virtual:   Whether to include virtual events.
        top_k:             Number of results to return.
        date_weight:       Weight for date proximity score (0-1).
        similarity_weight: Weight for semantic similarity score (0-1).

    Returns:
        List of RankedEvent objects sorted by final score descending.
    """
    index = _get_index()

    # Build metadata filters
    filters = {}
    if industry:            filters["industry"]          = {"$eq": industry}
    if location:            filters["location"]          = {"$eq": location}

    results = index.search(
        namespace=NAMESPACE,
        query={
            "inputs": {"text": query},
            "top_k": top_k * 2,
            **({"filter": filters} if filters else {}),
        }
    )

    ranked = []
    for match in results.get("result", {}).get("hits", []):
        fields  = match.get("fields", {})
        sim     = match.get("_score", 0)
        d_score = _date_score(fields.get("date", ""))
        final   = round((similarity_weight * sim) + (date_weight * d_score), 4)

        ranked.append(RankedEvent(
            title=               fields.get("title", ""),
            date=                fields.get("date", ""),
            location=            fields.get("location", ""),
            venue=               fields.get("venue"),
            is_virtual=          fields.get("is_virtual", False),
            industry=            fields.get("industry", ""),
            event_type=          fields.get("event_type"),
            experience_level=    fields.get("experience_level"),
            companies_attending= fields.get("companies_attending"),
            target_majors=       fields.get("target_majors"),
            organizer=           fields.get("organizer"),
            description=         fields.get("description", ""),
            registration_url=    fields.get("registration_url", ""),
            source=              fields.get("source", ""),
            score=               final,
            similarity_score=    round(sim, 4),
            date_score=          round(d_score, 4),
        ))

    return sorted(ranked, key=lambda e: e.score, reverse=True)[:top_k]
