# api/main.py  — PitWall F1 Predictor backend  v2.1
import os, time, logging, asyncio, warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional
from functools import wraps
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request # Add Request to your existing fastapi imports
import fastf1
import numpy as np
import pandas as pd
import xgboost as xgb
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sys
from pipeline.post_race import run_post_race

# suppress pandas chained-assignment warnings globally
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.options.mode.chained_assignment = None

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Paste this immediately after app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows your laptop to talk to the Pi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Paths ──────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "model",  "model.json")
CACHE_DIR    = os.path.join(BASE_DIR, "data",   "cache")
FEATURES_CSV = os.path.join(BASE_DIR, "data",   "features.csv")
RAW_CSV      = os.path.join(BASE_DIR, "data",   "raw_races.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("pitwall")

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Feature config ─────────────────────────────────────────────
FEATURE_COLS = [
    "QualiGapToPole","GridPosition","DriverFormLast3","TeamFormLast3",
    "DNF_last3","CircuitHistory","StartingTyreEncoded","IsWet",
    "SprintPosition","TrackType",
]
TYRE_MAP  = {"SOFT":0,"MEDIUM":1,"HARD":2,"INTER":3,"WET":4}
TRACK_MAP = {"power":0,"balanced":1,"high_downforce":2,"street":3,"high_deg":4}

CIRCUIT_TRACK_TYPE = {
    "Bahrain Grand Prix":"high_deg","Saudi Arabian Grand Prix":"street",
    "Australian Grand Prix":"balanced","Japanese Grand Prix":"high_downforce",
    "Chinese Grand Prix":"balanced","Miami Grand Prix":"street",
    "Emilia Romagna Grand Prix":"balanced","Monaco Grand Prix":"street",
    "Spanish Grand Prix":"balanced","Canadian Grand Prix":"balanced",
    "Austrian Grand Prix":"power","British Grand Prix":"balanced",
    "Hungarian Grand Prix":"high_downforce","Belgian Grand Prix":"power",
    "Dutch Grand Prix":"high_downforce","Italian Grand Prix":"power",
    "Azerbaijan Grand Prix":"street","Singapore Grand Prix":"street",
    "United States Grand Prix":"high_deg","Mexico City Grand Prix":"high_downforce",
    "São Paulo Grand Prix":"balanced","Las Vegas Grand Prix":"street",
    "Qatar Grand Prix":"high_deg","Abu Dhabi Grand Prix":"high_deg",
}

FLAG_MAP = {
    "Bahrain":"🇧🇭","Saudi Arabia":"🇸🇦","Australia":"🇦🇺","Japan":"🇯🇵",
    "China":"🇨🇳","United States":"🇺🇸","Italy":"🇮🇹","Monaco":"🇲🇨",
    "Spain":"🇪🇸","Canada":"🇨🇦","Austria":"🇦🇹","Great Britain":"🇬🇧",
    "Hungary":"🇭🇺","Belgium":"🇧🇪","Netherlands":"🇳🇱","Azerbaijan":"🇦🇿",
    "Singapore":"🇸🇬","Mexico":"🇲🇽","Brazil":"🇧🇷","Qatar":"🇶🇦",
    "Abu Dhabi":"🇦🇪","United Arab Emirates":"🇦🇪",
}

IST = timezone(timedelta(hours=5, minutes=30))

def fmt_ist(dt) -> str:
    """Cross-platform IST datetime formatting (works on Windows + Linux)."""
    ist = pd.to_datetime(dt, utc=True).astimezone(IST)
    day   = ist.strftime("%a")
    date  = str(ist.day)          # no leading zero, cross-platform
    month = ist.strftime("%b")
    hour  = ist.hour % 12 or 12   # 12-hour, no leading zero
    minute= ist.strftime("%M")
    ampm  = "AM" if ist.hour < 12 else "PM"
    return f"{day} {date} {month} · {hour}:{minute} {ampm}"

# ── Auth ───────────────────────────────────────────────────────
API_KEY = os.getenv("PITWALL_API_KEY", "changeme-set-env-var")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ── Rate limiter ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── TTL cache ──────────────────────────────────────────────────
_cache: dict = {}

def cached(ttl_seconds: int):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            key = fn.__name__
            now = time.time()
            if key in _cache:
                value, expires = _cache[key]
                if now < expires:
                    return value
            result = await fn(*args, **kwargs)
            _cache[key] = (result, now + ttl_seconds)
            return result
        return wrapper
    return decorator

def invalidate(key: str): _cache.pop(key, None)

# ── App state ──────────────────────────────────────────────────
class AppState:
    model:           Optional[xgb.XGBRegressor] = None
    last_prediction: Optional[dict]             = None
    live_lap:        Optional[int]              = None
    live_session:    Optional[str]              = None
    live_event:      Optional[str]              = None
    mid_race_done:   bool                       = False
    scheduler_active:bool                       = False
    last_poll:       Optional[datetime]         = None
    last_update:     Optional[datetime]         = None
    race_end_detected:  bool = False
    post_race_done:     bool = False   # guard: run once per race
    last_lap_count:     int  = 0
    stale_lap_polls:    int  = 0       # consecutive polls with no new lap  

state = AppState()

def load_model() -> xgb.XGBRegressor:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model.json not found: {MODEL_PATH}")
    m = xgb.XGBRegressor()
    m.load_model(MODEL_PATH)
    log.info("Model loaded: %s", MODEL_PATH)
    return m

# ── Pydantic schemas ───────────────────────────────────────────
class DriverInput(BaseModel):
    driver:            str
    team:              str
    grid_position:     int
    quali_gap_to_pole: float
    driver_form_last3: float
    team_form_last3:   float
    dnf_last3:         int   = 0
    circuit_history:   float = 10.0
    starting_tyre:     str   = "MEDIUM"
    sprint_position:   int   = 10
    is_wet:            int   = 0
    track_type:        str   = "balanced"

class PredictRequest(BaseModel):
    race:    str
    year:    int
    round:   int
    drivers: list[DriverInput]

class PredictResult(BaseModel):
    position:   int
    driver:     str
    team:       str
    confidence: float

class PredictResponse(BaseModel):
    race:         str
    predicted_at: str
    source:       str
    results:      list[PredictResult]

class UpdateResponse(BaseModel):
    status:  str
    message: str

class LiveStatusResponse(BaseModel):
    scheduler_active: bool
    live_event:       Optional[str]
    live_session:     Optional[str]
    live_lap:         Optional[int]
    mid_race_done:    bool
    last_poll:        Optional[str]
    last_update:      Optional[str]
    last_prediction:  Optional[dict]

# ── Prediction core ────────────────────────────────────────────
def _build_row(d: DriverInput) -> dict:
    return {
        "QualiGapToPole":      d.quali_gap_to_pole,
        "GridPosition":        d.grid_position,
        "DriverFormLast3":     d.driver_form_last3,
        "TeamFormLast3":       d.team_form_last3,
        "DNF_last3":           d.dnf_last3,
        "CircuitHistory":      d.circuit_history,
        "StartingTyreEncoded": TYRE_MAP.get(d.starting_tyre.upper(), 1),
        "IsWet":               d.is_wet,
        "SprintPosition":      d.sprint_position,
        "TrackType":           TRACK_MAP.get(d.track_type.lower(), 1),
    }

def run_prediction(request: PredictRequest, source: str = "quali_input") -> PredictResponse:
    rows       = [_build_row(d) for d in request.drivers]
    df         = pd.DataFrame(rows, columns=FEATURE_COLS)
    raw_scores = state.model.predict(df)
    ranks      = np.argsort(np.argsort(raw_scores)) + 1
    inv        = raw_scores.max() - raw_scores
    conf       = inv / (inv.max() + 1e-9)

    results = sorted([
        PredictResult(
            position   = int(ranks[i]),
            driver     = d.driver,
            team       = d.team,
            confidence = round(float(conf[i]), 3),
        )
        for i, d in enumerate(request.drivers)
    ], key=lambda r: r.position)

    return PredictResponse(
        race         = request.race,
        predicted_at = datetime.now(timezone.utc).isoformat(),
        source       = source,
        results      = results,
    )

# ══════════════════════════════════════════════════════════════
# Auto-data endpoints
# ══════════════════════════════════════════════════════════════

@cached(ttl_seconds=21600)
async def _get_next_race_data() -> dict:
    now  = datetime.now(timezone.utc)
    year = now.year

    # Use .copy() to avoid SettingWithCopyWarning
    sched = fastf1.get_event_schedule(year, include_testing=False).copy()
    sched["EventDate"] = pd.to_datetime(sched["EventDate"], utc=True)

    future = sched[sched["EventDate"] >= now - pd.Timedelta(hours=4)]
    if future.empty:
        return {"error": "No upcoming races found"}

    event      = future.iloc[0]
    event_name = str(event["EventName"])
    round_num  = int(event["RoundNumber"])
    total      = int(sched[sched["RoundNumber"] > 0]["RoundNumber"].max())

    sessions = []
    for key, name in [("Session1","Practice 1"),("Session2","Practice 2"),
                      ("Session3","Practice 3"),("Session4","Qualifying"),
                      ("Session5","Race")]:
        date_key = key + "Date"
        if date_key in event.index and pd.notna(event[date_key]):
            sessions.append({
                "name":    name,
                "ist":     fmt_ist(event[date_key]),
                "is_race": name == "Race",
            })

    race_utc = None
    if "Session5Date" in event.index and pd.notna(event["Session5Date"]):
        race_utc = pd.to_datetime(event["Session5Date"], utc=True).isoformat()

    race_ist_label = ""
    if race_utc:
        race_ist_label = "Race: " + fmt_ist(race_utc) + " IST"

    country  = str(event.get("Country", ""))
    flag     = FLAG_MAP.get(country, "🏁")
    location = f"{event.get('Location', '')}, {country}"

    return {
        "name":         event_name,
        "flag":         flag,
        "location":     location,
        "round":        round_num,
        "total_rounds": total,
        "race_utc":     race_utc,
        "race_ist":     race_ist_label,
        "sessions":     sessions,
        "track_type":   CIRCUIT_TRACK_TYPE.get(event_name, "balanced"),
    }


@cached(ttl_seconds=86400)
async def _get_last_race_data() -> dict:
    if not os.path.exists(RAW_CSV):
        return {"error": "raw_races.csv not found"}

    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["FinalPosition"]).copy()
    df["FinalPosition"] = df["FinalPosition"].astype(int)
    df = df.sort_values(["Year","RoundNumber"])

    latest = df.iloc[-1]
    year   = int(latest["Year"])
    rnd    = int(latest["RoundNumber"])
    event  = str(latest["EventName"])

    race_df = df[(df["Year"]==year)&(df["RoundNumber"]==rnd)].sort_values("FinalPosition")

    def parse_time(t):
        if not pd.notna(t): return "DNF"
        s = str(t)
        if "days" not in s: return s
        try:
            td   = pd.to_timedelta(s)
            secs = int(td.total_seconds())
            h = secs//3600; m=(secs%3600)//60; sec=secs%60
            return f"{h}:{m:02d}:{sec:02d}"
        except Exception:
            return "—"

    podium = [
        {"pos":int(r["FinalPosition"]),"abbr":str(r["Abbreviation"]),
         "team":str(r["TeamName"]),"time":parse_time(r.get("Time"))}
        for _,r in race_df[race_df["FinalPosition"]<=3].iterrows()
    ]

    results = [
        {"pos":int(r["FinalPosition"]),"abbr":str(r["Abbreviation"]),
         "team":str(r["TeamName"]),
         "pts":int(r["Points"]) if pd.notna(r.get("Points")) else 0,
         "status":str(r["Status"])}
        for _,r in race_df[race_df["FinalPosition"]>3].head(10).iterrows()
    ]

    return {"name":event,"year":year,"round":rnd,
            "sub":f"Round {rnd} · {year}","podium":podium,"results":results}


@cached(ttl_seconds=86400)
async def _get_accuracy_data() -> dict:
    if not os.path.exists(FEATURES_CSV) or state.model is None:
        return {"winner":0,"podium":0,"mae":0,"races":0,"season":2026}

    from sklearn.metrics import mean_absolute_error

    df          = pd.read_csv(FEATURES_CSV)
    latest_year = int(df["Year"].max())
    season      = df[df["Year"]==latest_year].copy()

    if season.empty:
        return {"winner":0,"podium":0,"mae":0,"races":0,"season":latest_year}

    X = season[FEATURE_COLS].fillna(10)
    season["PredScore"] = state.model.predict(X)
    season["PredRank"]  = season.groupby(["Year","RoundNumber"])["PredScore"].rank()

    mae            = mean_absolute_error(season["FinalPosition"], season["PredRank"])
    winner_correct = 0
    podium_correct = 0
    total_races    = 0

    for _, grp in season.groupby(["Year","RoundNumber"]):
        aw = set(grp[grp["FinalPosition"]==1]["Abbreviation"])
        pw = set(grp[grp["PredRank"]==1]["Abbreviation"])
        ap = set(grp[grp["FinalPosition"]<=3]["Abbreviation"])
        pp = set(grp[grp["PredRank"]<=3]["Abbreviation"])
        if aw & pw: winner_correct += 1
        podium_correct += len(ap & pp)
        total_races    += 1

    return {
        "winner": round(winner_correct/total_races*100,1) if total_races else 0,
        "podium": round(podium_correct/(total_races*3)*100,1) if total_races else 0,
        "mae":    round(mae,2),
        "races":  total_races,
        "season": latest_year,
    }


@cached(ttl_seconds=3600)
async def _get_rough_prediction() -> dict:
    if not os.path.exists(FEATURES_CSV) or state.model is None:
        return {"results":[],"race":"Unknown","source":"form_only"}

    df = pd.read_csv(FEATURES_CSV)

    try:
        next_race  = await _get_next_race_data()
        race_name  = next_race.get("name","Next Race")
        track_type = TRACK_MAP.get(next_race.get("track_type","balanced"),1)
    except Exception:
        race_name  = "Next Race"
        track_type = 1

    latest_year = int(df["Year"].max())
    latest_rnd  = int(df[df["Year"]==latest_year]["RoundNumber"].max())
    recent      = df[(df["Year"]==latest_year)&(df["RoundNumber"]==latest_rnd)].copy()

    if recent.empty:
        return {"results":[],"race":race_name,"source":"form_only"}

    rows = [{
        "QualiGapToPole":      5.0,
        "GridPosition":        10.0,
        "DriverFormLast3":     float(row.get("DriverFormLast3",10)),
        "TeamFormLast3":       float(row.get("TeamFormLast3",10)),
        "DNF_last3":           float(row.get("DNF_last3",0)),
        "CircuitHistory":      float(row.get("CircuitHistory",10)),
        "StartingTyreEncoded": 1.0,
        "IsWet":               0.0,
        "SprintPosition":      10.0,
        "TrackType":           float(track_type),
    } for _,row in recent.iterrows()]

    feat_df    = pd.DataFrame(rows, columns=FEATURE_COLS)
    raw_scores = state.model.predict(feat_df)
    ranks      = np.argsort(np.argsort(raw_scores)) + 1
    inv        = raw_scores.max() - raw_scores
    conf       = inv / (inv.max() + 1e-9)

    results = sorted([{
        "position":   int(ranks[i]),
        "driver":     str(row["Abbreviation"]),
        "team":       str(row["TeamName"]),
        "confidence": round(float(conf[i]),3),
    } for i,(_, row) in enumerate(recent.iterrows())], key=lambda r:r["position"])

    return {
        "race":    race_name,
        "source":  "form_only",
        "note":    f"Based on {latest_year} Round {latest_rnd} form. Sharpens after qualifying.",
        "results": results[:10],
    }

# ── Live lap-18 re-predict ─────────────────────────────────────
def _fetch_live_and_repredict(year:int, round_num:int, race_name:str):
    log.info("Lap-18 update: %s %d", race_name, year)
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)

        laps  = session.laps
        lap18 = laps[laps["LapNumber"] <= 18]
        if lap18.empty:
            log.warning("No lap-18 data yet")
            return

        latest = (
            lap18.sort_values("LapNumber").groupby("Driver").last()
            .reset_index()[["Driver","Position","Compound"]]
        )

        weather = session.weather_data
        is_wet  = int(weather["Rainfall"].iloc[-1]>0) \
                  if weather is not None and not weather.empty else 0

        if state.last_prediction is None:
            return

        prior      = state.last_prediction["_request"]
        driver_map = {r["Driver"]:r for _,r in latest.iterrows()}

        patched = []
        for d in prior["drivers"]:
            abbr = d["driver"]
            if abbr in driver_map:
                row = driver_map[abbr]
                if not pd.isna(row["Position"]):
                    d["grid_position"] = int(row["Position"])
                if row["Compound"] in TYRE_MAP:
                    d["starting_tyre"] = str(row["Compound"])
                d["is_wet"] = is_wet
            patched.append(DriverInput(**d))

        req      = PredictRequest(race=prior["race"],year=prior["year"],
                                  round=prior["round"],drivers=patched)
        response = run_prediction(req, source="live_lap18")
        state.last_prediction             = response.model_dump()
        state.last_prediction["_request"] = prior
        state.last_update                 = datetime.now(timezone.utc)
        state.mid_race_done               = True
        invalidate("_get_rough_prediction")
        log.info("Lap-18 done: %s", race_name)

    except Exception as e:
        log.error("Lap-18 failed: %s", e)

# ── Scheduler ──────────────────────────────────────────────────
STALE_POLLS_THRESHOLD = 3   # 3 polls * 3 min = 9 min of no new lap = race over

async def _poll_session():
    try:
        now   = datetime.now(timezone.utc)
        year  = now.year

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        schedule["EventDate"] = pd.to_datetime(schedule["EventDate"], utc=True)

        window = schedule[
            (schedule["EventDate"] >= now - pd.Timedelta(days=4)) &
            (schedule["EventDate"] <= now + pd.Timedelta(days=4))
        ]

        if window.empty:
            state.scheduler_active = False
            state.live_session     = None
            state.live_event       = None
            state.live_lap         = None
            state.last_poll        = now
            return

        event      = window.iloc[0]
        round_num  = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        state.live_event       = event_name
        state.scheduler_active = True
        state.last_poll        = now

        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps

            if laps.empty:
                state.live_session = "Pre-Race / Quali"
                state.live_lap     = None
                return

            current_lap       = int(laps["LapNumber"].max())
            state.live_lap    = current_lap
            state.live_session = "Race"

            # ── Race-end detection ─────────────────────────────────────────
            if current_lap == state.last_lap_count:
                state.stale_lap_polls += 1
            else:
                state.stale_lap_polls = 0
                state.last_lap_count  = current_lap

            race_ended = (
                state.stale_lap_polls >= STALE_POLLS_THRESHOLD
                and current_lap > 40   # ignore early session gaps
                and not state.post_race_done
            )

            if race_ended:
                log.info("Race end detected for %s at lap %d — running post-race pipeline", event_name, current_lap)
                state.post_race_done = True   # set immediately to prevent duplicate runs
                loop = asyncio.get_event_loop()
                prior = state.last_prediction.get("_request") if state.last_prediction else None
                await loop.run_in_executor(
                    None,
                    lambda: run_post_race(year, round_num, event_name, prior_predict_request=prior)
                )

            # ── Lap-18 trigger (unchanged) ─────────────────────────────────
            if current_lap >= 18 and not state.mid_race_done:
                log.info("Lap 18 reached — triggering live re-prediction")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    _fetch_live_and_repredict,
                    year, round_num, event_name,
                )

        except Exception:
            state.live_session = "Session not live yet"
            state.live_lap     = None

    except Exception as e:
        log.error("Scheduler poll error: %s", e)


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    state.model = load_model()
    scheduler.add_job(_poll_session, "interval", minutes=3, id="poll")
    scheduler.start()
    log.info("Scheduler started")
    yield
    scheduler.shutdown(wait=False)

# ── App setup ──────────────────────────────────────────────────
app = FastAPI(title="PitWall API", version="2.1.0", lifespan=lifespan,
              docs_url="/docs", redoc_url=None)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

scheduler = AsyncIOScheduler(timezone="UTC")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,"
    "http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,"
    "null"   # allows opening index.html directly from filesystem
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = False,
    allow_methods     = ["GET","POST"],
    allow_headers     = ["Content-Type","X-Api-Key"],
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]         = "DENY"
    response.headers["X-XSS-Protection"]        = "1; mode=block"
    response.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
    return response

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    return {"status":"ok","model_loaded":state.model is not None,
            "scheduler":state.scheduler_active}

@app.get("/next-race")
@limiter.limit("20/minute")
async def next_race(request: Request):
    try:    return await _get_next_race_data()
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/last-race")
@limiter.limit("20/minute")
async def last_race(request: Request):
    try:    return await _get_last_race_data()
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/accuracy")
@limiter.limit("20/minute")
async def accuracy(request: Request):
    try:    return await _get_accuracy_data()
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/rough-prediction")
@limiter.limit("10/minute")
async def rough_prediction(request: Request):
    try:    return await _get_rough_prediction()
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, payload: PredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    response = run_prediction(payload, source="quali_input")
    state.last_prediction             = response.model_dump()
    state.last_prediction["_request"] = payload.model_dump()
    state.mid_race_done               = False
    state.last_update                 = datetime.now(timezone.utc)
    return response

@app.post("/update")
@limiter.limit("5/minute")
async def manual_update(request: Request, _: str = Depends(verify_api_key)):
    if state.last_prediction is None:
        raise HTTPException(400, "No prediction in state — call /predict first")
    prior = state.last_prediction["_request"]
    loop  = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _fetch_live_and_repredict,
        prior["year"], prior["round"], prior["race"],
    )
    return UpdateResponse(status="ok", message=f"Updated: {prior['race']}")

@app.get("/live-status")
@limiter.limit("30/minute")
async def live_status(request: Request):
    pred_public = {k:v for k,v in state.last_prediction.items()
                   if not k.startswith("_")} if state.last_prediction else None
    return LiveStatusResponse(
        scheduler_active = state.scheduler_active,
        live_event       = state.live_event,
        live_session     = state.live_session,
        live_lap         = state.live_lap,
        mid_race_done    = state.mid_race_done,
        last_poll        = state.last_poll.isoformat()   if state.last_poll   else None,
        last_update      = state.last_update.isoformat() if state.last_update else None,
        last_prediction  = pred_public,
    )
@app.post("/retrain", dependencies=[Depends(verify_api_key)])
async def retrain():
    """
    Admin: manually trigger a full retrain on the current features.csv.
    Does NOT fetch new race data — use /post-race for that first.
    """
    if state.last_prediction is None:
        raise HTTPException(status_code=400, detail="No race context in state")

    prior     = state.last_prediction.get("_request", {})
    year      = prior.get("year", datetime.now().year)
    round_num = prior.get("round", 0)
    race      = prior.get("race", "unknown")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_post_race(year, round_num, race, prior_predict_request=prior, retrain=True)
    )

    # Reload model into state after retrain
    state.model = load_model()
    return result
@app.get("/accuracy")
def accuracy():
    """
    Returns the full accuracy log across all logged races.
    Used by the frontend accuracy dashboard.
    """
    from pipeline.post_race import _load_accuracy_log
    log_data = _load_accuracy_log()
    if not log_data:
        return {"races": [], "summary": {}}

    winner_acc  = sum(1 for r in log_data if r["winner_correct"]) / len(log_data)
    podium_acc  = sum(r["podium_acc"] for r in log_data) / len(log_data)
    avg_mae     = sum(r["mae"] for r in log_data) / len(log_data)

    return {
        "races": log_data,
        "summary": {
            "total_races":   len(log_data),
            "winner_acc":    round(winner_acc, 3),
            "podium_acc":    round(podium_acc, 3),
            "avg_mae":       round(avg_mae, 3),
        }
    }
@app.post("/post-race", dependencies=[Depends(verify_api_key)])
async def post_race(retrain: bool = False):
    if state.last_prediction is None:
        raise HTTPException(400, "No prediction in state — call /predict first")
    prior = state.last_prediction.get("_request")
    loop  = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_post_race(
            prior["year"], prior["round"], prior["race"],
            prior_predict_request=prior,
            retrain=retrain,
        )
    )
    return result