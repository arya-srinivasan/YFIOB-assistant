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
    venue: Optional[str] = None
    is_virtual: bool = False
    registration_url: Optional[str] = None

    organizer: Optional[str] = None
    organizer_type: Optional[str] = None
    industry: Optional[str] = None
    event_type: Optional[str] = None
    experience_level: Optional[str] = None
    target_majors: Optional[str] = None
    companies_attending: Optional[list[str]] = None