import logging
from apscheduler.schedulers.background import BackgroundScheduler
from .tools.google_search_tool import google_search_career_events
from .tools.pinecone_tool import ingest_events_to_pinecone

logger = logging.getLogger(__name__)

def _refresh(location: str, industry: str | None):
    logger.info(f"[Scheduler] refreshing events for '{location}' ...")
    try:
        google_events = google_search_career_events(location=location, industry=industry)
        result = ingest_events_to_pinecone(google_events)
        logger.info(f"[Scheduler] Done - upserted: {result['upserted']} | failed: {result['failed']}")
    except Exception as e:
        logger.error(f"[Scheduler] Refresh failed: {e}")

def start_scheduler(
    location: str,
    industry: str | None = None,
    interval_minutes: int = 20,
) -> BackgroundScheduler:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=_refresh,
        trigger="interval",
        minutes=interval_minutes,
        kwargs={"location": location, "industry": industry},
        max_instances=1,
    )
    scheduler.start()
    logger.info(f"[Scheduler] Started — every {interval_minutes}min for '{location}'")

    _refresh(location, industry)
    return scheduler