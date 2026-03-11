from pydantic import BaseModel
from typing import Optional

class CareerEvent(BaseModel):
    id: str
    source: str
    scraped_at: str

    title: str
    description: Optional[str] = None
    date: Optional[str] = None
    end_date: Optional[str] = None
    location: str
    event_type: str = "general"
    venue: Optional[str] = None
    registration_url: Optional[str] = None

    organizer: Optional[str] = None
    industry: str = "General"
    target_majors: Optional[list[str]] = None