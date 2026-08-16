from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_agent, resume_travel_agent

# This is to allow nested event loops for async calls in FastAPI
import nest_asyncio
nest_asyncio.apply()


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    action: str
    feedback: str | None = None


def _plan_response(result: dict):
    """
    Shared response body for /api/travel and
    /api/travel/approve so the frontend only has to
    understand one shape.
    """

    return {
        "success": True,
        "status": result["status"],
        "thread_id": result["thread_id"],
        "answer": result["answer"],
        "draft_itinerary": result.get("draft_itinerary", ""),
        "rejection_reason": result.get("rejection_reason", ""),
        "revision_count": result.get("revision_count", 0),
        "flight_results": result["flight_results"],
        "hotel_results": result["hotel_results"],
        "weather_results": result["weather_results"],
        "itinerary": result["itinerary"],
        "llm_calls": result["llm_calls"],
    }



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        return JSONResponse(
            content=_plan_response(result)
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.post("/api/travel/approve")
async def approve_plan(request_data: ApprovalRequest):
    """
    Human-in-the-loop endpoint.

    Resumes a plan paused at the approval gate, either
    approving the draft itinerary or sending it back
    with feedback.
    """

    try:
        thread_id = request_data.thread_id.strip()
        action = request_data.action.strip().lower()

        if not thread_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "thread_id is required."
                }
            )

        if action not in ("approve", "revise"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "action must be 'approve' or 'revise'."
                }
            )

        feedback = (request_data.feedback or "").strip()

        if action == "revise" and not feedback:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Feedback is required to request changes."
                }
            )

        result = resume_travel_agent(
            thread_id=thread_id,
            action=action,
            feedback=feedback
        )

        return JSONResponse(
            content=_plan_response(result)
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )



@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )