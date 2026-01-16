# models.py
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base

def now():
    return datetime.utcnow()

class User(Base):
    __tablename__ = "users"
    tg_user_id = Column(Integer, primary_key=True, index=True)

    energy = Column(Integer, default=30)
    energy_max = Column(Integer, default=30)
    energy_updated_at = Column(DateTime, default=now)

    revive_cards = Column(Integer, default=0)
    undo_tokens = Column(Integer, default=0)
    shuffle_tokens = Column(Integer, default=0)

    last_video_cursor = Column(Integer, default=0)  # 用于轮播

class AdVideo(Base):
    __tablename__ = "ad_videos"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(Text, nullable=False)
    active = Column(Boolean, default=True)

class AdSession(Base):
    __tablename__ = "ad_sessions"
    ad_session_id = Column(String, primary_key=True, index=True)
    tg_user_id = Column(Integer, index=True)

    reward_type = Column(String, nullable=False)  # revive/undo/shuffle/energy
    reward_amount = Column(Integer, default=1)

    required_seconds = Column(Integer, default=10)
    watched_seconds = Column(Integer, default=0)

    created_at = Column(DateTime, default=now)
    expires_at = Column(DateTime, nullable=False)
    last_beat_at = Column(DateTime, nullable=True)
    claimed = Column(Boolean, default=False)

class AdClaim(Base):
    __tablename__ = "ad_claims"
    id = Column(Integer, primary_key=True)
    tg_user_id = Column(Integer, index=True)
    reward_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=now)

class GameRun(Base):
    """
    一次“羊了个羊式两关流程”叫一个 run：
    stage1 = easy
    stage2 = hell
    """
    __tablename__ = "game_runs"
    run_id = Column(String, primary_key=True, index=True)
    tg_user_id = Column(Integer, index=True)

    stage1_done = Column(Boolean, default=False)
    stage2_done = Column(Boolean, default=False)
    finished = Column(Boolean, default=False)

    # 计时与排行榜（服务器时间为准）
    started_at = Column(DateTime, default=now)
    stage1_finished_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    total_ms = Column(Integer, nullable=True)  # 第一关+第二关总耗时（毫秒）

    created_at = Column(DateTime, default=now)

class GameSession(Base):
    __tablename__ = "game_sessions"
    session_id = Column(String, primary_key=True, index=True)

    tg_user_id = Column(Integer, index=True)
    run_id = Column(String, ForeignKey("game_runs.run_id"), nullable=True)

    stage = Column(Integer, default=1)  # 1 or 2
    mode = Column(String, default="easy")  # easy/hell
    seed = Column(String, nullable=False)

    status = Column(String, default="playing")  # playing/failed/won/ended

    energy_cost = Column(Integer, default=1)

    # 你要求的：每关一次
    revive_used = Column(Boolean, default=False)
    undo_used = Column(Boolean, default=False)
    shuffle_used = Column(Boolean, default=False)

    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now)
