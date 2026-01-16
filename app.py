# app.py
import os, uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from db import engine, Base, get_db
from models import User, AdVideo, AdSession, AdClaim, GameRun, GameSession
from security import sign_token, require_dev_secret, get_current_user

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "dev_admin_token_change_me")

AD_REQUIRED_SECONDS = int(os.getenv("AD_REQUIRED_SECONDS", "10"))
AD_SESSION_EXPIRE_SECONDS = int(os.getenv("AD_SESSION_EXPIRE_SECONDS", "120"))
AD_CLAIM_MAX_PER_DAY = int(os.getenv("AD_CLAIM_MAX_PER_DAY", "20"))
AD_CLAIM_MIN_INTERVAL_SECONDS = int(os.getenv("AD_CLAIM_MIN_INTERVAL_SECONDS", "3"))
AD_CLAIM_REQUIRE_RECENT_BEAT_SECONDS = int(os.getenv("AD_CLAIM_REQUIRE_RECENT_BEAT_SECONDS", "3"))

# 体力恢复
ENERGY_REGEN_SECONDS = int(os.getenv("ENERGY_REGEN_SECONDS", "300"))  # 5分钟+1

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def _now():
    return datetime.utcnow()

def _dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    return int(dt.timestamp() * 1000)

async def _refresh_energy(u: User, db: AsyncSession):
    now = _now()
    last = u.energy_updated_at or now
    if u.energy >= u.energy_max:
        u.energy_updated_at = now
        return

    elapsed = (now - last).total_seconds()
    if elapsed <= 0:
        return
    gain = int(elapsed // ENERGY_REGEN_SECONDS)
    if gain <= 0:
        return

    u.energy = min(u.energy_max, u.energy + gain)
    u.energy_updated_at = last + timedelta(seconds=gain * ENERGY_REGEN_SECONDS)
    db.add(u)

def _energy_next_in(u: User) -> int:
    if u.energy >= u.energy_max:
        return 0
    now = _now()
    last = u.energy_updated_at or now
    elapsed = (now - last).total_seconds()
    remain = ENERGY_REGEN_SECONDS - int(elapsed % ENERGY_REGEN_SECONDS)
    return max(1, remain)

def me_payload(u: User):
    return {
        "tg_user_id": u.tg_user_id,
        "energy": u.energy,
        "energy_max": u.energy_max,
        "energy_next_in": _energy_next_in(u),
        "revive_cards": u.revive_cards,
        "undo_tokens": u.undo_tokens,
        "shuffle_tokens": u.shuffle_tokens,
    }

# -------------------------
# Auth
# -------------------------
@app.post("/auth/dev")
async def auth_dev(body: dict, db: AsyncSession = Depends(get_db)):
    tg_user_id = int(body.get("tg_user_id", 0))
    secret = str(body.get("secret", ""))
    require_dev_secret(secret)

    r = await db.execute(select(User).where(User.tg_user_id == tg_user_id))
    u = r.scalar_one_or_none()
    if not u:
        u = User(tg_user_id=tg_user_id)
        db.add(u)
        await db.commit()
        await db.refresh(u)

    await _refresh_energy(u, db)
    await db.commit()

    token = sign_token({"tg_user_id": tg_user_id, "exp": int(datetime.utcnow().timestamp()) + 3600*24})
    return {"access_token": token}

@app.get("/me")
async def me(u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _refresh_energy(u, db)
    await db.commit()
    return me_payload(u)

# -------------------------
# Admin: import videos
# -------------------------
@app.post("/admin/videos/import")
async def admin_import_videos(
    body: dict,
    x_admin_token: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="admin forbidden")

    urls = body.get("urls") or []
    if not isinstance(urls, list) or not urls:
        raise HTTPException(status_code=400, detail="urls required")

    count = 0
    for url in urls:
        if not url or not isinstance(url, str):
            continue
        db.add(AdVideo(url=url.strip(), active=True))
        count += 1
    await db.commit()
    return {"ok": True, "count": count}

async def _pick_next_video(u: User, db: AsyncSession) -> str:
    r = await db.execute(select(AdVideo).where(AdVideo.active == True).order_by(AdVideo.id.asc()))
    vids = r.scalars().all()
    if not vids:
        raise HTTPException(status_code=400, detail="no active videos")
    cursor = u.last_video_cursor or 0
    idx = cursor % len(vids)
    u.last_video_cursor = cursor + 1
    db.add(u)
    return vids[idx].url

# -------------------------
# Ads
# -------------------------
def reward_amount(reward_type: str) -> int:
    if reward_type == "energy":
        return 5
    return 1

@app.post("/ad/start")
async def ad_start(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    reward_type = (body.get("reward_type") or "revive").strip().lower()
    if reward_type not in ("revive", "undo", "shuffle", "energy"):
        raise HTTPException(status_code=400, detail="bad reward_type")

    await _refresh_energy(u, db)
    video_url = await _pick_next_video(u, db)

    sid = uuid.uuid4().hex
    expires_at = _now() + timedelta(seconds=AD_SESSION_EXPIRE_SECONDS)

    sess = AdSession(
        ad_session_id=sid,
        tg_user_id=u.tg_user_id,
        reward_type=reward_type,
        reward_amount=reward_amount(reward_type),
        required_seconds=AD_REQUIRED_SECONDS,
        watched_seconds=0,
        expires_at=expires_at,
        last_beat_at=None,
        claimed=False,
    )
    db.add(sess)
    await db.commit()

    return {
        "ad_session_id": sid,
        "required_seconds": AD_REQUIRED_SECONDS,
        "video_url": video_url,
        "reward_type": reward_type,
        "reward_amount": sess.reward_amount,
    }

@app.post("/ad/heartbeat")
async def ad_heartbeat(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sid = body.get("ad_session_id")
    visible = bool(body.get("visible", False))
    playing = bool(body.get("playing", False))
    if not sid:
        raise HTTPException(status_code=400, detail="missing ad_session_id")

    r = await db.execute(select(AdSession).where(AdSession.ad_session_id == sid))
    sess = r.scalar_one_or_none()
    if not sess or sess.tg_user_id != u.tg_user_id:
        raise HTTPException(status_code=404, detail="ad session not found")
    if sess.claimed:
        return {"watched_seconds": sess.watched_seconds, "required": sess.required_seconds}
    if sess.expires_at < _now():
        raise HTTPException(status_code=400, detail="ad session expired")

    now = _now()
    if sess.last_beat_at:
        delta = (now - sess.last_beat_at).total_seconds()
        if delta < 0.6:
            return {"watched_seconds": sess.watched_seconds, "required": sess.required_seconds}

    sess.last_beat_at = now
    if visible and playing and sess.watched_seconds < sess.required_seconds:
        sess.watched_seconds += 1

    db.add(sess)
    await db.commit()
    return {"watched_seconds": sess.watched_seconds, "required": sess.required_seconds}

@app.post("/ad/claim")
async def ad_claim(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sid = body.get("ad_session_id")
    if not sid:
        raise HTTPException(status_code=400, detail="missing ad_session_id")

    r = await db.execute(select(AdSession).where(AdSession.ad_session_id == sid))
    sess = r.scalar_one_or_none()
    if not sess or sess.tg_user_id != u.tg_user_id:
        raise HTTPException(status_code=404, detail="ad session not found")

    now = _now()
    if sess.claimed:
        await _refresh_energy(u, db)
        await db.commit()
        return {"ok": True, "reward_type": sess.reward_type, "reward_amount": 0, "me": me_payload(u)}

    if sess.expires_at < now:
        raise HTTPException(status_code=400, detail="ad session expired")
    if sess.watched_seconds < sess.required_seconds:
        raise HTTPException(status_code=400, detail="not completed (watch more)")
    if sess.last_beat_at and (now - sess.last_beat_at).total_seconds() > AD_CLAIM_REQUIRE_RECENT_BEAT_SECONDS:
        raise HTTPException(status_code=400, detail="not completed (lost heartbeat)")

    today = datetime.utcnow().date()
    start = datetime(today.year, today.month, today.day)
    end = start + timedelta(days=1)

    rr = await db.execute(
        select(func.count(AdClaim.id)).where(
            AdClaim.tg_user_id == u.tg_user_id,
            AdClaim.created_at >= start,
            AdClaim.created_at < end,
        )
    )
    used_today = int(rr.scalar() or 0)
    if used_today >= AD_CLAIM_MAX_PER_DAY:
        raise HTTPException(status_code=400, detail="daily limit reached")

    rr2 = await db.execute(
        select(AdClaim.created_at).where(
            AdClaim.tg_user_id == u.tg_user_id
        ).order_by(AdClaim.created_at.desc()).limit(1)
    )
    last = rr2.scalar_one_or_none()
    if last and (now - last).total_seconds() < AD_CLAIM_MIN_INTERVAL_SECONDS:
        raise HTTPException(status_code=400, detail="too frequent")

    rt = sess.reward_type
    amt = sess.reward_amount

    await _refresh_energy(u, db)

    if rt == "revive":
        u.revive_cards += amt
    elif rt == "undo":
        u.undo_tokens += amt
    elif rt == "shuffle":
        u.shuffle_tokens += amt
    elif rt == "energy":
        u.energy = min(u.energy_max, u.energy + amt)

    sess.claimed = True
    db.add(AdClaim(tg_user_id=u.tg_user_id, reward_type=rt))
    db.add(u)
    db.add(sess)
    await db.commit()

    return {"ok": True, "reward_type": rt, "reward_amount": amt, "me": me_payload(u)}

# -------------------------
# Game
# -------------------------
def make_seed(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"

async def _get_session(db: AsyncSession, u: User, session_id: str) -> GameSession:
    r = await db.execute(select(GameSession).where(GameSession.session_id == session_id))
    sess = r.scalar_one_or_none()
    if not sess or sess.tg_user_id != u.tg_user_id:
        raise HTTPException(status_code=404, detail="session not found")
    return sess

@app.post("/game/start_run")
async def game_start_run(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _refresh_energy(u, db)

    cost = 1
    if u.energy < cost:
        raise HTTPException(status_code=400, detail="not enough energy")
    u.energy -= cost

    run_id = uuid.uuid4().hex
    run = GameRun(run_id=run_id, tg_user_id=u.tg_user_id, started_at=_now())
    db.add(run)

    session_id = uuid.uuid4().hex
    seed = make_seed("easy")

    sess = GameSession(
        session_id=session_id,
        tg_user_id=u.tg_user_id,
        run_id=run_id,
        stage=1,
        mode="easy",
        seed=seed,
        status="playing",
        energy_cost=cost,
    )
    db.add(sess)
    db.add(u)
    await db.commit()

    return {
        "run_id": run_id,
        "session_id": session_id,
        "stage": 1,
        "mode": "easy",
        "seed": seed,
        "energy_cost": cost,
        "run_started_at_ms": _dt_to_ms(run.started_at),
        "me": me_payload(u),
        "flags": {"revive_used": False, "undo_used": False, "shuffle_used": False},
    }

@app.post("/game/next_stage")
async def game_next_stage(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="missing run_id")

    r = await db.execute(select(GameRun).where(GameRun.run_id == run_id))
    run = r.scalar_one_or_none()
    if not run or run.tg_user_id != u.tg_user_id:
        raise HTTPException(status_code=404, detail="run not found")
    if not run.stage1_done:
        raise HTTPException(status_code=400, detail="stage1 not finished")
    if run.finished or run.stage2_done:
        raise HTTPException(status_code=400, detail="run already finished")

    session_id = uuid.uuid4().hex
    seed = make_seed("hell")

    sess = GameSession(
        session_id=session_id,
        tg_user_id=u.tg_user_id,
        run_id=run_id,
        stage=2,
        mode="hell",
        seed=seed,
        status="playing",
        energy_cost=0,
    )
    db.add(sess)
    await db.commit()

    return {
        "run_id": run_id,
        "session_id": session_id,
        "stage": 2,
        "mode": "hell",
        "seed": seed,
        "energy_cost": 0,
        "run_started_at_ms": _dt_to_ms(run.started_at),
        "me": me_payload(u),
        "flags": {"revive_used": False, "undo_used": False, "shuffle_used": False},
    }

@app.post("/game/restart_level")
async def game_restart_level(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    真·重开本关：生成一个新的 session_id，清空本关一次性道具使用标记。
    seed 不变（你要“最终结果不变”的那种味儿）。
    """
    sid = body.get("session_id")
    if not sid:
        raise HTTPException(status_code=400, detail="missing session_id")

    old = await _get_session(db, u, sid)

    # 结束旧 session，避免你误用
    old.status = "ended"
    old.updated_at = _now()
    db.add(old)

    new_sid = uuid.uuid4().hex
    new_sess = GameSession(
        session_id=new_sid,
        tg_user_id=old.tg_user_id,
        run_id=old.run_id,
        stage=old.stage,
        mode=old.mode,
        seed=old.seed,
        status="playing",
        energy_cost=0,
        revive_used=False,
        undo_used=False,
        shuffle_used=False,
    )
    db.add(new_sess)
    await db.commit()

    return {
        "ok": True,
        "session_id": new_sid,
        "run_id": old.run_id,
        "stage": old.stage,
        "mode": old.mode,
        "seed": old.seed,
        "flags": {"revive_used": False, "undo_used": False, "shuffle_used": False},
        "me": me_payload(u),
    }

@app.post("/game/fail")
async def game_fail(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sid = body.get("session_id")
    sess = await _get_session(db, u, sid)
    if sess.status != "playing":
        return {"ok": True, "me": me_payload(u), "status": sess.status}

    sess.status = "failed"
    sess.updated_at = _now()
    db.add(sess)
    await db.commit()
    return {"ok": True, "me": me_payload(u), "status": sess.status}

@app.post("/game/finish")
async def game_finish(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sid = body.get("session_id")
    result = (body.get("result") or "").lower()
    sess = await _get_session(db, u, sid)

    if result not in ("win", "lose"):
        raise HTTPException(status_code=400, detail="bad result")

    sess.status = "won" if result == "win" else "ended"
    sess.updated_at = _now()
    db.add(sess)

    total_ms = None

    if sess.run_id:
        rr = await db.execute(select(GameRun).where(GameRun.run_id == sess.run_id))
        run = rr.scalar_one_or_none()
        if run and run.tg_user_id == u.tg_user_id:
            if sess.stage == 1 and result == "win":
                run.stage1_done = True
                if not run.stage1_finished_at:
                    run.stage1_finished_at = _now()

            if sess.stage == 2 and result == "win":
                run.stage2_done = True
                run.finished = True
                if not run.finished_at:
                    run.finished_at = _now()
                if run.started_at and run.finished_at:
                    total_ms = int((run.finished_at - run.started_at).total_seconds() * 1000)
                    run.total_ms = total_ms

            db.add(run)

    await db.commit()
    return {"ok": True, "total_ms": total_ms}


@app.get("/leaderboard")
async def leaderboard(
    limit: int = 20,
    day: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """排行榜：按“每个用户的最好成绩”排序。

    - limit: 返回前 N 名
    - day: 可选，YYYY-MM-DD（UTC）只看当天成绩；不传则看全量
    """
    limit = max(1, min(int(limit or 20), 100))

    filters = [GameRun.finished == True, GameRun.total_ms.is_not(None)]
    if day:
        try:
            d = datetime.strptime(day, "%Y-%m-%d").date()
            start = datetime(d.year, d.month, d.day)
            end = start + timedelta(days=1)
            filters.append(GameRun.finished_at >= start)
            filters.append(GameRun.finished_at < end)
        except Exception:
            raise HTTPException(status_code=400, detail="bad day format, want YYYY-MM-DD")

    subq = (
        select(
            GameRun.tg_user_id.label("tg_user_id"),
            func.min(GameRun.total_ms).label("best_ms"),
        )
        .where(*filters)
        .group_by(GameRun.tg_user_id)
        .subquery()
    )

    r = await db.execute(
        select(subq.c.tg_user_id, subq.c.best_ms)
        .order_by(subq.c.best_ms.asc())
        .limit(limit)
    )
    rows = r.all()
    items = []
    for i, (tg, ms) in enumerate(rows, start=1):
        items.append({"rank": i, "tg_user_id": tg, "total_ms": int(ms)})

    return {"ok": True, "items": items}


@app.get("/leaderboard/me")
async def leaderboard_me(
    day: Optional[str] = None,
    u: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """返回当前用户的最好成绩与排名（按最好成绩计算）。"""

    filters = [GameRun.finished == True, GameRun.total_ms.is_not(None)]
    if day:
        try:
            d = datetime.strptime(day, "%Y-%m-%d").date()
            start = datetime(d.year, d.month, d.day)
            end = start + timedelta(days=1)
            filters.append(GameRun.finished_at >= start)
            filters.append(GameRun.finished_at < end)
        except Exception:
            raise HTTPException(status_code=400, detail="bad day format, want YYYY-MM-DD")

    best_r = await db.execute(
        select(func.min(GameRun.total_ms))
        .where(*filters)
        .where(GameRun.tg_user_id == u.tg_user_id)
    )
    best_ms = best_r.scalar_one_or_none()
    if best_ms is None:
        return {"ok": True, "has_score": False}

    subq = (
        select(
            GameRun.tg_user_id.label("tg_user_id"),
            func.min(GameRun.total_ms).label("best_ms"),
        )
        .where(*filters)
        .group_by(GameRun.tg_user_id)
        .subquery()
    )
    rank_r = await db.execute(select(func.count()).select_from(subq).where(subq.c.best_ms < int(best_ms)))
    better = int(rank_r.scalar() or 0)
    return {
        "ok": True,
        "has_score": True,
        "best_ms": int(best_ms),
        "rank": better + 1,
    }

@app.post("/game/revive")
async def game_revive(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sid = body.get("session_id")
    sess = await _get_session(db, u, sid)

    if sess.status != "failed":
        raise HTTPException(status_code=400, detail="not in failed state")

    if sess.revive_used:
        raise HTTPException(status_code=400, detail="revive already used in this level")

    if u.revive_cards <= 0:
        raise HTTPException(status_code=400, detail="no revive cards")

    u.revive_cards -= 1
    sess.revive_used = True
    sess.status = "playing"
    sess.updated_at = _now()

    db.add(u)
    db.add(sess)
    await db.commit()

    return {
        "ok": True,
        "revive_cost": 1,
        "me": me_payload(u),
        "flags": {"revive_used": True, "undo_used": sess.undo_used, "shuffle_used": sess.shuffle_used},
    }

@app.post("/game/use_item")
async def game_use_item(body: dict, u: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sid = body.get("session_id")
    item = (body.get("item") or "").lower()
    if item not in ("undo", "shuffle"):
        raise HTTPException(status_code=400, detail="bad item")

    sess = await _get_session(db, u, sid)
    if sess.status != "playing":
        raise HTTPException(status_code=400, detail="not playing")

    if item == "undo":
        if sess.undo_used:
            raise HTTPException(status_code=400, detail="undo already used in this level")
        if u.undo_tokens <= 0:
            raise HTTPException(status_code=400, detail="no undo tokens")
        u.undo_tokens -= 1
        sess.undo_used = True

    if item == "shuffle":
        if sess.shuffle_used:
            raise HTTPException(status_code=400, detail="shuffle already used in this level")
        if u.shuffle_tokens <= 0:
            raise HTTPException(status_code=400, detail="no shuffle tokens")
        u.shuffle_tokens -= 1
        sess.shuffle_used = True

    sess.updated_at = _now()
    db.add(u)
    db.add(sess)
    await db.commit()

    return {
        "ok": True,
        "me": me_payload(u),
        "flags": {"revive_used": sess.revive_used, "undo_used": sess.undo_used, "shuffle_used": sess.shuffle_used},
    }
