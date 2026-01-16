# security.py
import os, time, hmac, hashlib, base64, json
from fastapi import Header, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db import get_db
from models import User

JWT_SECRET = os.getenv("JWT_SECRET", "dev_jwt_secret_change_me").encode("utf-8")
DEV_MODE = os.getenv("DEV_MODE", "1") == "1"
DEV_AUTH_SECRET = os.getenv("DEV_AUTH_SECRET", "dev_login_secret")

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _b64url_dec(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))

def sign_token(payload: dict) -> str:
    header = {"alg":"HS256","typ":"JWT"}
    h = _b64url(json.dumps(header, separators=(",",":")).encode())
    p = _b64url(json.dumps(payload, separators=(",",":")).encode())
    msg = f"{h}.{p}".encode("utf-8")
    sig = hmac.new(JWT_SECRET, msg, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"

def verify_token(token: str) -> dict:
    try:
        h, p, s = token.split(".")
        msg = f"{h}.{p}".encode("utf-8")
        sig = _b64url_dec(s)
        exp_sig = hmac.new(JWT_SECRET, msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, exp_sig):
            raise ValueError("bad sig")
        payload = json.loads(_b64url_dec(p))
        if payload.get("exp", 0) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")

async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing token")
    token = authorization.split(" ", 1)[1].strip()
    payload = verify_token(token)
    tg_user_id = int(payload["tg_user_id"])

    r = await db.execute(select(User).where(User.tg_user_id == tg_user_id))
    u = r.scalar_one_or_none()
    if not u:
        u = User(tg_user_id=tg_user_id)
        db.add(u)
        await db.commit()
        await db.refresh(u)
    return u

def require_dev_secret(secret: str):
    if not DEV_MODE:
        raise HTTPException(status_code=403, detail="dev auth disabled")
    if secret != DEV_AUTH_SECRET:
        raise HTTPException(status_code=403, detail="bad dev secret")
