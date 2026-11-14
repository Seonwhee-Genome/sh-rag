"""HTTP routes for the client workspace: /board and /conversations.

Payloads and status codes follow docs/client-api.md.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from ingest.workspace import Conflict, NotFound, Workspace

# --- bodies -------------------------------------------------------------------


class CardCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    notes: str = ""
    column: str = "backlog"
    priority: str = "medium"
    conversation_id: str | None = None
    position: str = "bottom"


class CardPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = None
    column: str | None = None
    priority: str | None = None
    conversation_id: str | None = None


class CardMove(BaseModel):
    column: str
    before: str | None = None
    after: str | None = None


class ConversationCreate(BaseModel):
    title: str = Field(default="", max_length=200)
    model: str = ""


class ConversationPatch(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    model: str | None = None


class MessageCreate(BaseModel):
    role: str
    content: str = ""
    sources: list[dict[str, Any]] | None = None
    error: str | None = None


class MessagePatch(BaseModel):
    content: str | None = None
    sources: list[dict[str, Any]] | None = None
    error: str | None = None


def _call(fn: Callable, *args, **kwargs):
    """Map workspace errors onto HTTP statuses."""
    try:
        return fn(*args, **kwargs)
    except NotFound as exc:
        raise HTTPException(404, str(exc))
    except Conflict as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


def make_router(workspace: Workspace) -> APIRouter:
    router = APIRouter()

    # --- board ----------------------------------------------------------------

    @router.get("/board", tags=["board"])
    def get_board() -> dict:
        return workspace.board()

    @router.post("/board/cards", status_code=201, tags=["board"])
    def create_card(body: CardCreate) -> dict:
        return _call(workspace.create_card, **body.model_dump())

    @router.get("/board/cards/{card_id}", tags=["board"])
    def get_card(card_id: str) -> dict:
        return _call(workspace.get_card, card_id)

    @router.patch("/board/cards/{card_id}", tags=["board"])
    def patch_card(card_id: str, body: CardPatch) -> dict:
        changes = body.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(422, "no fields to update")
        return _call(workspace.update_card, card_id, **changes)

    @router.post("/board/cards/{card_id}/move", tags=["board"])
    def move_card(card_id: str, body: CardMove) -> dict:
        return _call(workspace.move_card, card_id, body.column, before=body.before, after=body.after)

    @router.delete("/board/cards/{card_id}", status_code=204, tags=["board"])
    def delete_card(card_id: str) -> Response:
        _call(workspace.delete_card, card_id)
        return Response(status_code=204)

    # --- conversations ----------------------------------------------------------

    @router.get("/conversations", tags=["conversations"])
    def list_conversations(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict:
        return workspace.list_conversations(limit=limit, offset=offset)

    @router.post("/conversations", status_code=201, tags=["conversations"])
    def create_conversation(body: ConversationCreate) -> dict:
        return _call(workspace.create_conversation, title=body.title, model=body.model)

    @router.get("/conversations/{cid}", tags=["conversations"])
    def get_conversation(cid: str) -> dict:
        return _call(workspace.get_conversation, cid)

    @router.patch("/conversations/{cid}", tags=["conversations"])
    def patch_conversation(cid: str, body: ConversationPatch) -> dict:
        changes = body.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(422, "no fields to update")
        return _call(workspace.update_conversation, cid, **changes)

    @router.delete("/conversations/{cid}", status_code=204, tags=["conversations"])
    def delete_conversation(cid: str) -> Response:
        _call(workspace.delete_conversation, cid)
        return Response(status_code=204)

    @router.post("/conversations/{cid}/messages", status_code=201, tags=["conversations"])
    def add_message(cid: str, body: MessageCreate) -> dict:
        return _call(workspace.add_message, cid, body.role, body.content, sources=body.sources, error=body.error)

    @router.patch("/conversations/{cid}/messages/{mid}", tags=["conversations"])
    def patch_message(cid: str, mid: str, body: MessagePatch) -> dict:
        changes = body.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(422, "no fields to update")
        return _call(workspace.update_message, cid, mid, **changes)

    @router.delete("/conversations/{cid}/messages/{mid}", status_code=204, tags=["conversations"])
    def delete_message(cid: str, mid: str) -> Response:
        _call(workspace.delete_message, cid, mid)
        return Response(status_code=204)

    return router
