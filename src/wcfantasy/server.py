"""Local web server (FastAPI) for the wcfantasy advisor.

`python -m wcfantasy serve` starts this on 127.0.0.1 and opens the browser.
It exposes the same routines as the CLI through a small JSON API backed by
the service layer, and serves the static single-page app in `web/`. The slow
commands (advise / fetch / odds / calibrate) run as background jobs
(`jobs.py`) the frontend polls, so the UI never blocks. Nothing here writes
to the FIFA account: it advises and records local state, exactly like the CLI.
"""

import threading
import webbrowser
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import jobs, service

DEFAULT_PORT = 8765
WEB_DIR = Path(__file__).resolve().parent / "web"
INDEX_FILE = WEB_DIR / "index.html"


class ApplyBody(BaseModel):
    force: bool = False


class BoosterBody(BaseModel):
    name: str | None = None


class RecordBody(BaseModel):
    captain: str | None = None
    sub_out: str | None = None
    sub_in: str | None = None
    vice: str | None = None
    mark_changed: bool = False


class CalibrateBody(BaseModel):
    write: bool = False


class SettingsBody(BaseModel):
    language: str


def create_app() -> FastAPI:
    """Build the FastAPI app (separate from `serve` so tests can drive it)."""
    app = FastAPI(title="wcfantasy", docs_url="/api/docs", openapi_url="/api/openapi.json")

    # The SPA assets are iterated on constantly; without this the browser's
    # heuristic-freshness cache serves stale CSS/JS (StaticFiles sends only
    # ETag/Last-Modified, no Cache-Control). On localhost the files are tiny,
    # so just never cache them: every load gets the current bytes.
    @app.middleware("http")
    async def _no_store(request, call_next):
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/assets"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---- read-only -------------------------------------------------------
    @app.get("/api/team")
    def get_team() -> dict:
        return service.team_snapshot()

    @app.get("/api/status")
    def get_status() -> dict:
        return service.status_line()

    @app.get("/api/settings")
    def get_settings() -> dict:
        return service.get_settings()

    @app.post("/api/settings")
    def post_settings(body: SettingsBody) -> dict:
        try:
            return service.set_settings(body.language)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ---- background jobs -------------------------------------------------
    @app.post("/api/advise")
    def post_advise() -> dict:
        return {"job_id": jobs.start(service.advise_plan, kind="advise")}

    @app.post("/api/advise/from-scratch")
    def post_advise_from_scratch() -> dict:
        job_id = jobs.start(
            lambda log: service.advise_plan(from_scratch=True, log=log),
            kind="advise_from_scratch",
        )
        return {"job_id": job_id}

    @app.get("/api/live")
    def get_live() -> dict:
        return {"job_id": jobs.start(service.live_state, kind="live")}

    @app.get("/api/odds")
    def get_odds() -> dict:
        return {"job_id": jobs.start(service.odds_table, kind="odds")}

    @app.get("/api/bracket")
    def get_bracket() -> dict:
        return {"job_id": jobs.start(service.predicted_bracket, kind="bracket")}

    @app.post("/api/fetch")
    def post_fetch() -> dict:
        return {"job_id": jobs.start(service.fetch_feeds, kind="fetch")}

    @app.post("/api/calibrate")
    def post_calibrate(body: CalibrateBody = Body(default=CalibrateBody())) -> dict:
        job_id = jobs.start(
            lambda log: service.calibrate_preview(write=body.write, log=log),
            kind="calibrate",
        )
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        snap = jobs.get(job_id)
        if snap is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return snap

    # ---- sync state changes ---------------------------------------------
    @app.post("/api/apply")
    def post_apply(body: ApplyBody = Body(default=ApplyBody())) -> dict:
        return service.apply_advice(force=body.force)

    @app.post("/api/booster")
    def post_booster(body: BoosterBody) -> dict:
        return service.set_booster(body.name)

    @app.post("/api/live/record")
    def post_record(body: RecordBody) -> dict:
        return service.record_live(
            captain=body.captain, sub_out=body.sub_out, sub_in=body.sub_in,
            vice=body.vice, mark_changed=body.mark_changed,
        )

    # ---- static SPA ------------------------------------------------------
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    if (WEB_DIR / "assets").is_dir():
        app.mount(
            "/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets"
        )

    return app


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the web server on 127.0.0.1 and (optionally) open the browser."""
    import uvicorn

    url = f"http://127.0.0.1:{port}"
    if open_browser:
        # Open after a short delay so the server is accepting connections.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"wcfantasy web app running on {url}  (press Ctrl+C to stop)")
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
