"""Change feed for clients: GET /events as Server-Sent Events.

Jobs are watched by snapshot (they change often while running); the board and
conversations announce a new version whenever the workspace writes. One stream
replaces polling in native clients (see docs/client-api.md, "GET /events").
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncIterator

TOPICS = ("jobs", "board", "conversations")
POLL_SECONDS = 0.5
PING_SECONDS = 15.0


class ChangeFeed:
    """Version counters per topic, bumped by the workspace on every write."""

    def __init__(self):
        self._lock = threading.Lock()
        self._versions = {"board": 0, "conversations": 0}
        self._last_id: dict[str, str | None] = {"board": None, "conversations": None}

    def bump(self, topic: str, id_: str | None = None) -> None:
        with self._lock:
            self._versions[topic] = self._versions.get(topic, 0) + 1
            self._last_id[topic] = id_

    def snapshot(self) -> dict[str, tuple[int, str | None]]:
        with self._lock:
            return {t: (self._versions[t], self._last_id[t]) for t in self._versions}


def _event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def parse_topics(raw: str | None) -> set[str]:
    if not raw:
        return set(TOPICS)
    topics = {t.strip() for t in raw.split(",") if t.strip()}
    unknown = topics - set(TOPICS)
    if unknown:
        raise ValueError(f"unknown topics: {', '.join(sorted(unknown))}; valid: {', '.join(TOPICS)}")
    return topics


async def event_stream(jobs, workspace, feed: ChangeFeed, topics: set[str],
                       poll: float = POLL_SECONDS, ping: float = PING_SECONDS) -> AsyncIterator[str]:
    """Yield SSE frames until the client disconnects.

    The first frames describe the current state (running jobs, current versions),
    so a client can treat connecting as a refresh.
    """
    job_state: dict[str, str] = {}
    if "jobs" in topics:
        for job in jobs.list():
            job_state[job["id"]] = json.dumps(job, sort_keys=True)
            if job["status"] == "running":
                yield _event("job", job)
    versions = feed.snapshot()
    if "board" in topics:
        yield _event("board", {"version": workspace.version("board")})
    if "conversations" in topics:
        yield _event("conversations", {"version": workspace.version("conversations"), "id": None})

    since_ping = 0.0
    while True:
        await asyncio.sleep(poll)
        since_ping += poll
        if "jobs" in topics:
            current = {}
            for job in jobs.list():
                encoded = json.dumps(job, sort_keys=True)
                current[job["id"]] = encoded
                if job_state.get(job["id"]) != encoded:
                    yield _event("job", job)
            job_state = current
        now = feed.snapshot()
        for topic in ("board", "conversations"):
            if topic in topics and now[topic] != versions[topic]:
                data = {"version": workspace.version(topic)}
                if topic == "conversations":
                    data["id"] = now[topic][1]
                yield _event(topic, data)
        versions = now
        if since_ping >= ping:
            since_ping = 0.0
            yield _event("ping", {})
