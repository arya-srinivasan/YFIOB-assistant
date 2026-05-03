# pip install fastmcp httpx

import httpx
from fastmcp import FastMCP
import os

ONET_API_KEY = os.getenv("ONET_API_KEY")
BASE_URL = "https://services.onetcenter.org"

mcp = FastMCP("onet")

def get_client():
    return httpx.AsyncClient(
        headers={"X-API-Key": ONET_API_KEY}
    )

@mcp.tool()
async def get_occupation_skills(onet_code: str, display: str = "long") -> dict:
    async with get_client() as client:
        r = await client.get(
            f"{BASEURL}/occupation/{onet_code}/skills",
            params={"display": display}
        )
        return r.json()

@mcp.tool()
async def get_occupation_salary(occupation: str, location: str = "long") -> dict:
    async with get_client() as client:
        r = await client.get(
            f"{BASEURL}/occupations/{onet_code}/outlook"
        )
        return r.json()

@mcp.tool()
async def get_interest_profiler_questions(onet_code: str, display: str = "long") -> dict:
    async with get_client() as client:
        r = await client.get(
            f"{BASEURL}/mnm/interestprofiler/questions"
        )
        return r.json()

@mcp.tool()
async def score_interest_profiler(answers: list[int]) -> dict:
    async with get_client() as client:
        r = await client.get(
            f"{BASEURL}/mnm/interestprofiler/results",
            params={"answers": ",".join(str(a) for a in answers)}
        )
        return r.json()

@mcp.tool()
async def get_careers_by_interest(riasec_scores: str, job_zone: int = 0) -> dict:
    async with get_client() as client:
        r = await client.get(
            f"{BASEURL}/mnm/interestprofiler/results",
            params={"scores": riasec_scores, "job_zone": job_zone}
        )
        return r.json()
    
#search careers, get career report, get bright outlook careers
@mcp.tool()
async def search_careers(keyword: str) -> dict:
    """Search O*NET for occupations matching a keyword or interest."""
    async with get_client() as client:
        r = await client.get(
            f"{BASE_URL}/mnm/search",
            params={"keyword": keyword, "end": 5}
        )
        data = r.json()
        return data.get("occupation", [])

@mcp.tool()
async def get_career_report(onet_code: str) -> dict:
    """Get a full career report for an occupation — tasks, work environment, outlook."""
    async with get_client() as client:
        r = await client.get(
            f"{BASE_URL}/mnm/careers/{onet_code}/report"
        )
        data   = r.json()
        career = data.get("career", {})
        return {
            "title":          career.get("title"),
            "what_they_do":   career.get("what_they_do"),
            "on_the_job":     data.get("on_the_job", {}),
            "outlook":        data.get("job_outlook", {}),
            "bright_outlook": career.get("bright_outlook", False),
        }

@mcp.tool()
async def get_bright_outlook_careers(keyword: str = "") -> dict:
    """Get fast-growing and in-demand careers. Optional keyword to filter."""
    async with get_client() as client:
        params = {"bright_outlook": "true", "end": 8}
        if keyword:
            params["keyword"] = keyword
        r = await client.get(f"{BASE_URL}/mnm/search", params=params)
        return r.json().get("occupation", [])