from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from env import ChaosOpsEnv


app = FastAPI(title="ChaosOps", version="1.0.0")
env = ChaosOpsEnv()
UI_FILE = os.path.join(os.path.dirname(__file__), "ui.html")


class ResetRequest(BaseModel):
    task: str = Field(default="task3")


class StepRequest(BaseModel):
    action: str
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "ChaosOps",
        "message": "Use POST /reset and POST /step",
        "ui": "/ui",
        "docs": "/docs",
    }


@app.get("/ui")
def ui_page() -> FileResponse:
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "status": "healthy"}


@app.post("/reset")
def reset_endpoint(request: ResetRequest) -> Dict[str, Any]:
    try:
        return env.reset(task_name=request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"unexpected error: {exc}") from exc


@app.post("/step")
def step_endpoint(request: StepRequest) -> Dict[str, Any]:
    try:
        return env.step(action=request.action, payload=request.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"unexpected error: {exc}") from exc
