# sh-rag client API contract

**API version 2.** The contract that native clients (such as `sh-rag-client` for iOS) and
the web console build on. Everything here is implemented by the stack in this repository;
anything not listed is internal and may change without notice.

Two HTTP services are involved:

| Service | Default URL | Role |
|---|---|---|
| **Gateway** (Go) | `http://localhost:8090` | Chat. OpenAI-compatible, stateless, streams answers with retrieved sources. |
| **Ingest / console API** (Python) | `http://localhost:8001` | Everything else: status and stats, documents and files, models, jobs, retrieval settings, the **board** and **conversations** store, and a change feed. |

Clients discover what a server supports from `GET /healthz` (gateway) and `GET /status`
(ingest), see [Feature discovery](#feature-discovery).

---

## Conventions

- **JSON** request and response bodies, `snake_case` keys. Unknown fields in responses must
  be ignored by clients; fields are only ever added within an API version.
- **Timestamps** are RFC 3339 strings in UTC (`2026-11-14T06:31:42.123456+00:00`). Clients
  must accept any number of fractional digits and either `Z` or `+00:00`.
- **IDs**: cards, conversations and messages use UUID strings. Jobs use 12 hex characters.
  Documents are addressed by `doc_id` (UUID) **or** by `source` (path relative to the
  documents folder, e.g. `uploads/report.pdf`).
- **Errors**
  - Gateway: `{"error": {"message": "...", "type": "sh_rag_error"}}`
  - Ingest: FastAPI style, `{"detail": "..."}` or `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` for validation (HTTP 422).
- **Authentication**
  - Gateway: `Authorization: Bearer <GATEWAY_API_KEY>` on every `/v1/*` route. `/healthz` is open.
  - Ingest: optional. When `INGEST_API_KEY` is set in `.env`, every route except `/`, `/health`,
    `/docs`, `/redoc` and `/openapi.json` requires `Authorization: Bearer <key>`. The key is also
    accepted as an `X-API-Key` header or an `api_key` query parameter (for `/files/...` links and
    `/events`, where headers cannot be set). Unset means open, which is the right default for
    `localhost` only.
- **Reaching the stack from a device**: ports are published on `127.0.0.1` by default. Set
  `BIND_ADDRESS=0.0.0.0` in `.env` to publish the gateway, ingest API and OpenWebUI on all
  interfaces, and set both `GATEWAY_API_KEY` and `INGEST_API_KEY` before doing so.
- **Streams** (`/v1/chat/completions` with `stream: true` and `/events`) are
  `text/event-stream`. Each event is `data: <json>\n\n`; the change feed also sets `event:`.
- **Concurrency** on the board and conversations is last-write-wins. Each resource carries
  `updated_at`, and each collection exposes a monotonically increasing `version` so clients
  can refetch cheaply.

### Feature discovery

```http
GET /healthz                       (gateway, no auth)
{"status": "ok", "version": "0.3.0", "features": ["sources", "retrieve_v2", "models_meta"]}

GET /status                        (ingest)
{..., "version": "0.3.0", "auth_required": false,
 "api": {"version": 2, "features": ["stats", "documents_detail", "files", "events",
                                    "job_cancel", "model_suggest", "board", "conversations"]}}
```

A client should check for a feature before relying on it and fall back (for example, to
footer parsing of sources) when it is absent, so it keeps working against an older server.

---

## Gateway

### `GET /v1/models`

Chat models, one per installed Ollama chat model. Embedding models are excluded. The `id`
carries the `rag/` prefix and is what `POST /v1/chat/completions` expects. `sh_rag` adds
display metadata so clients do not need a second call to the ingest API.

```json
{"object": "list", "data": [
  {"id": "rag/qwen2.5:7b", "object": "model", "created": 1794292195, "owned_by": "sh-rag",
   "sh_rag": {"name": "qwen2.5:7b", "family": "qwen2", "parameter_size": "7.6B",
              "quantization": "Q4_K_M", "size": 4683087332}}
]}
```

### `POST /v1/chat/completions`

Standard OpenAI chat completion, streaming or not. The gateway embeds the **last user
message**, searches the active index, injects the passages as a system message and forwards
the request to Ollama. One extension object, `sh_rag`, is accepted in the request and is
removed before forwarding:

```json
{
  "model": "rag/qwen2.5:7b",
  "stream": true,
  "messages": [{"role": "user", "content": "What does the contract say about termination?"}],
  "sh_rag": {
    "sources": "json",
    "retrieve": true,
    "top_k": 5,
    "score_threshold": 0.35
  }
}
```

| Field | Default | Meaning |
|---|---|---|
| `sources` | `"footer"` (or `"none"` when `RAG_SHOW_SOURCES=false`) | `"footer"`: append a markdown **Sources** list to the answer text (what OpenWebUI shows). `"json"`: no footer; sources are returned structurally as described below. `"none"`: neither. |
| `retrieve` | `true` | `false` skips retrieval and answers with the plain model. |
| `top_k` | runtime setting | Passages to retrieve for this request only. |
| `score_threshold` | runtime setting | Minimum cosine similarity (0 to 1) for this request only. |

**Native clients should use `"sources": "json"`.** Parsing the footer is a compatibility
path for OpenAI-only clients and is not a stable format.

#### Structured sources (`"sources": "json"`)

A **Source** object:

```json
{"index": 1, "doc_id": "0d5b...", "source": "uploads/contract.pdf", "title": "Service Contract",
 "page": 4, "score": 0.7123, "text": "Either party may terminate ..."}
```

`index` is the bracketed number the model is asked to cite (`[1]`). `page` is omitted when
the format has no pages. `text` is the passage as it was given to the model.

A **Retrieval** object describes what was done:

```json
{"embed_model": "bge-m3", "collection": "documents_bge_m3", "top_k": 5, "score_threshold": 0.35,
 "took_ms": 118, "skipped": false}
```

`skipped` is `true` with a `reason` (`"disabled"`, `"no_query"`, `"task_prompt"`) when
retrieval did not run. If retrieval failed the gateway still answers without context and
sets `error` to the message (there will be no sources).

**Streaming.** The very first event, sent before any model output and before the first token
(so it arrives while a cold model is still loading), is a chunk with an empty delta and the
sources:

```
data: {"id": "chatcmpl-shrag-...", "object": "chat.completion.chunk", "created": 1794292300,
       "model": "qwen2.5:7b",
       "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
       "sh_rag": {"type": "sources", "sources": [Source, ...], "retrieval": Retrieval}}

data: {... normal Ollama chunks ...}

data: [DONE]
```

It is sent even when no passages matched (`"sources": []`), so a client can switch its
placeholder from "Searching your documents…" to "Answering…" the moment it arrives, and
render the sources list while the answer is still streaming.

**Non-streaming.** The response object gains a top-level `sh_rag`:

```json
{"id": "...", "object": "chat.completion", "choices": [...], "usage": {...},
 "sh_rag": {"sources": [Source, ...], "retrieval": Retrieval}}
```

**Errors.** Non-2xx responses from Ollama (e.g. unknown model, 404) are relayed with
Ollama's status and body. The client's "Retry" affordance should be offered for 5xx and
transport failures.

### `POST /v1/retrieve`

The exact retrieval the chat performs, without generation. Used by "test a search" screens.

```json
→ {"query": "termination notice period", "top_k": 5, "score_threshold": 0.3}
← {"chunks": [Source, ...], "retrieval": Retrieval}
```

`top_k` and `score_threshold` are optional and default to the runtime settings.

---

## Ingest / console API

Interactive OpenAPI docs for this service are served at `/docs`, and the machine-readable
schema at `/openapi.json`.

### Status and statistics

#### `GET /health`
`{"status": "ok"}`. Open, for liveness checks.

#### `GET /status`

```json
{"version": "0.3.0", "auth_required": false,
 "api": {"version": 2, "features": [...]},
 "embed_model": "bge-m3", "collection": "documents_bge_m3",
 "docs_path": "./data/docs", "scan_interval": 30, "chunk_size": 1000, "chunk_overlap": 150,
 "webui_url": "http://localhost:3000",
 "ollama": {"ok": true, "version": "0.6.5", "url": "http://ollama:11434"},
 "qdrant": {"ok": true}}
```

`ollama` / `qdrant` carry `{"ok": false, "error": "..."}` when unreachable. `webui_url` is
the OpenWebUI address as configured on the server; clients should link to it rather than
guessing a port.

#### `GET /stats`

Aggregates for dashboards, so a client does not have to download the document list to show
tiles and charts.

```json
{"documents": 12, "chunks": 340,
 "by_type": [{"type": "pdf", "documents": 7, "chunks": 260}, {"type": "md", "documents": 5, "chunks": 80}],
 "indexes": [{"collection": "documents_bge_m3", "active": true, "points": 340},
             {"collection": "documents_nomic_embed_text", "active": false, "points": 338}],
 "embed_model": "bge-m3",
 "jobs_running": 0,
 "workspace": {"cards": 9, "cards_by_column": {"backlog": 4, "in_progress": 2, "review": 1, "done": 2},
               "conversations": 14},
 "updated_at": "2026-11-14T06:31:42+00:00"}
```

`by_type` is sorted by document count, descending. `type` is the lower-cased file extension
without the dot, or `"other"`.

### Documents

#### `GET /documents`
```json
{"documents": [{"doc_id": "...", "source": "uploads/contract.pdf", "title": "Service Contract",
                "chunks": 42, "content_hash": "sha256...", "ingested_at": "..."}]}
```
Sorted by `source`.

#### `GET /documents/{key}`

`key` is a `doc_id` or a `source` path. Returns the document and its passages, in order.

```json
{"doc_id": "...", "source": "uploads/contract.pdf", "title": "Service Contract", "type": "pdf",
 "chunks": 42, "content_hash": "...", "ingested_at": "...",
 "size_bytes": 184233, "file_url": "/files/uploads/contract.pdf", "pages": 12,
 "passages": [{"index": 0, "page": 1, "text": "..."}, ...]}
```

`size_bytes` and `file_url` are `null` when the original file is no longer on disk. `pages`
is `null` for formats without pages. Pass `?include_text=false` to receive `passages` without
`text` (cheap outline for large files). 404 when unknown.

#### `GET /files/{source}`

The original file, served inline with its media type, for previewing a source a chat answer
cited (`Source.source` is exactly this path). 404 if the file is gone, 403 if the path
escapes the documents folder. Accepts `?api_key=` when ingest auth is on.

#### `POST /documents` (multipart, field `files`, repeatable)
Saves into `uploads/` and indexes immediately.
```json
{"summary": {"indexed": 2, "unsupported": 1},
 "results": [{"source": "uploads/a.pdf", "status": "indexed", "doc_id": "...", "chunks": 18, "error": null}, ...]}
```
`status` is one of `indexed | unchanged | empty | unsupported | removed | error`.

#### `DELETE /documents/{key}`
Removes from the index **and deletes the file**. Returns one result object as above with
`status: "removed"`. 404 when unknown.

#### `POST /reindex?force=false&background=true`
Sync the documents folder (`force=true` re-embeds everything). With `background=true` returns
a **Job** immediately (recommended for clients); otherwise blocks and returns the summary.

### Models

#### `GET /models`
```json
{"active_embed_model": "bge-m3",
 "models": [{"name": "qwen2.5:7b", "size": 4683087332, "modified_at": "...", "family": "qwen2",
             "parameter_size": "7.6B", "quantization": "Q4_K_M",
             "capabilities": ["completion", "tools"], "kind": "chat"}, ...]}
```
`kind` is `chat` or `embedding`. 502 when Ollama is unreachable.

#### `GET /models/suggest?q=qwen&kind=chat`
Autocomplete from the public Ollama library (cached, degrades to an empty list offline).
`kind` is optional (`chat` | `embedding`).
```json
{"models": [{"name": "qwen2.5", "kind": "chat", "capabilities": ["tools"], "sizes": ["0.5b", "7b", "72b"],
             "description": "..."}], "error": null}
```
Clients should offer these as the user types a model to pull; a `name` plus one of its
`sizes` (`qwen2.5:7b`) is a valid pull target.

#### `POST /models/pull` `{"model": "qwen2.5:7b"}` → Job
Download progress arrives on the job (`progress`, and `detail.completed` / `detail.total`
bytes). A cancelled pull keeps downloaded layers; pulling the same name again resumes.

#### `DELETE /models/{name}`
`{"deleted": "name"}`. 409 if it is the active embedding model, 404 if not installed.

#### `POST /embedding` `{"model": "nomic-embed-text"}` → Job
Switches the embedding model: pulls it if needed, embeds every document into a separate
collection, then activates it. Chat keeps the old index until the job is `done`. 400 if
already active, 409 if another switch is running.

### Retrieval settings and search

#### `GET /settings` · `PUT /settings`
```json
{"embed_model": "bge-m3", "top_k": 5, "score_threshold": 0.35}
```
`PUT` takes any subset of `top_k` (1–50) and `score_threshold` (0–1) and returns the full
object. The gateway picks changes up within a second. `embed_model` is changed only via
`POST /embedding`.

#### `POST /search`
```json
→ {"query": "...", "top_k": 5, "score_threshold": 0.3}
← {"results": [{"score": 0.71, "source": "uploads/contract.pdf", "page": 4, "text": "..."}]}
```
Same retrieval as the gateway's `/v1/retrieve`, available without the gateway key.

### Jobs

A **Job**:

```json
{"id": "a1b2c3d4e5f6", "kind": "pull", "key": "pull:qwen2.5:7b", "title": "Pull qwen2.5:7b",
 "status": "running", "message": "pulling 2bada8a74506", "progress": 0.42,
 "detail": {"model": "qwen2.5:7b", "completed": 1973741824, "total": 4683087332},
 "cancel_requested": false,
 "created_at": "...", "finished_at": null, "result": null, "error": null}
```

| Field | Notes |
|---|---|
| `kind` | `pull` · `reindex` · `embedding` |
| `status` | `running` · `done` · `error` · **`cancelled`**. Clients must render all four; a cancelled job is neither a success nor a failure. |
| `progress` | 0–1 or `null` when unknown (show an indeterminate bar). |
| `detail` | Kind-specific. Pulls: `model`, `completed`, `total` (bytes). Embedding switches: `model`. |
| `result` | On `done`: pulls `{"model": ...}`; reindex and embedding a status summary such as `{"indexed": 3, "removed": 1}`. |

Starting a job whose `key` is already running returns the existing job instead of a new one.

- `GET /jobs` → `{"jobs": [Job, ...]}`, newest first, last 30.
- `GET /jobs/{id}` → Job. 404 unknown.
- `POST /jobs/{id}/cancel` → Job. Asks it to stop at its next progress event. 409 if not running.

### `GET /events` — change feed (SSE)

One long-lived stream that replaces polling on the dashboard and the board. Optional
`?topics=jobs,board,conversations` (default: all).

```
event: job
data: {Job}                         // every time a job's state changes

event: board
data: {"version": 18}               // a card was created, changed, moved or deleted

event: conversations
data: {"version": 42, "id": "..."}  // a conversation or message changed; id may be null

event: ping
data: {}                            // every 15 s
```

Clients re-fetch the resource that changed (`GET /board`, `GET /conversations/{id}`), or
compare `version` with the one they hold. A client that cannot keep a stream open can poll
`GET /board` and `GET /conversations` and compare `version` instead.

---

## Board (kanban)

Server-side store for the client's board, so cards and their links to conversations survive
reinstalling the app and are shared between iPhone, iPad and the console.

**Columns** are fixed: `backlog`, `in_progress`, `review`, `done`.
**Priorities**: `low`, `medium`, `high`.

A **Card**:

```json
{"id": "9f3c...", "title": "Check termination clause", "notes": "Sources: contract.pdf (p. 4)",
 "column": "in_progress", "priority": "high", "position": 0,
 "conversation_id": "4b1e..." , "created_at": "...", "updated_at": "..."}
```

`position` is the 0-based order within its column. `conversation_id` optionally links the
card to a conversation (set when a card is created from an answer, or when "Ask sh-rag about
this card" starts a chat); it becomes `null` if the conversation is deleted.

| Method and path | Body | Returns |
|---|---|---|
| `GET /board` | – | `{"version": 18, "columns": [{"id": "backlog", "title": "Backlog"}, ...], "cards": [Card, ...]}` ordered by column then `position`. |
| `POST /board/cards` | `{"title", "notes"?, "column"?, "priority"?, "conversation_id"?, "position"?: "top" \| "bottom"}` | 201 Card. `title` is required (1–200 chars). Defaults: `backlog`, `medium`, `bottom`. |
| `PATCH /board/cards/{id}` | any of `title`, `notes`, `priority`, `conversation_id`, `column` | Card. Changing `column` here appends to the bottom of the new column; use **move** to place it. |
| `POST /board/cards/{id}/move` | `{"column", "before"?: card id, "after"?: card id}` | Card. Places the card in `column` before/after the given card (which must be in that column), or at the bottom when neither is given. Drag and drop maps directly onto this. |
| `DELETE /board/cards/{id}` | – | 204 |

Validation errors are 422; unknown card ids 404; `before`/`after` pointing to a card in a
different column 409.

---

## Conversations

Server-side chat history. The gateway stays stateless, so the **client writes the
transcript**: it posts the user's message, streams the answer from the gateway, then posts
the final assistant message together with the structured sources it received.

A **Conversation** (summary):

```json
{"id": "4b1e...", "title": "Termination clause", "model": "rag/qwen2.5:7b",
 "created_at": "...", "updated_at": "...", "message_count": 6,
 "preview": "The contract allows either party to terminate ..."}
```

A **Message**:

```json
{"id": "7c2a...", "conversation_id": "4b1e...", "role": "assistant",
 "content": "The contract allows ... [1]", "sources": [Source, ...] , "error": null,
 "created_at": "..."}
```

`role` is `system`, `user` or `assistant`. `sources` is `null` for user messages and for
assistant messages produced without retrieval. `error` holds a message when the answer
failed or was interrupted (`content` may then be empty or partial).

| Method and path | Body | Returns |
|---|---|---|
| `GET /conversations?limit=50&offset=0` | – | `{"version": 42, "total": 14, "conversations": [Conversation, ...]}` newest `updated_at` first. |
| `POST /conversations` | `{"title"?, "model"?}` | 201 Conversation. An empty title is filled from the first user message. |
| `GET /conversations/{id}` | – | Conversation plus `"messages": [Message, ...]` in order. |
| `PATCH /conversations/{id}` | `{"title"?, "model"?}` | Conversation |
| `DELETE /conversations/{id}` | – | 204. Cards linked to it keep their text and lose the link. |
| `POST /conversations/{id}/messages` | `{"role", "content", "sources"?, "error"?}` | 201 Message. Bumps the conversation's `updated_at`. |
| `PATCH /conversations/{id}/messages/{mid}` | any of `content`, `sources`, `error` | Message. Use it to finalize a message that was created before streaming finished. |
| `DELETE /conversations/{id}/messages/{mid}` | – | 204. Used by "retry" to drop a failed answer. |

### Recommended client flow for one question

1. `POST /conversations/{id}/messages` with the user's text (`role: "user"`).
2. `POST /v1/chat/completions` on the gateway with the whole transcript and
   `"sh_rag": {"sources": "json"}`, streaming. Show "Searching your documents…" until the first
   event (the sources chunk) arrives, then show the sources and stream the tokens.
3. On completion, `POST /conversations/{id}/messages` with `role: "assistant"`, the full
   `content` and the `sources` array from step 2. On failure, post it with `error` set so the
   transcript records what happened; "Retry" deletes that message and the user message
   before it, then repeats from step 1.

Clients may also keep a local copy for offline reading; `version` on `GET /conversations` tells
them when to resync.

---

## What the iOS client should change

Mapped from the screens in `sh-rag-client`:

| Screen | Today | With this contract |
|---|---|---|
| Chat answer | Splits `---\n**Sources**` out of the markdown. | Send `sh_rag.sources = "json"`, read the first stream event, render `Source` objects; tap a source → `GET /files/{source}` (QuickLook) or `GET /documents/{key}` for the passage list. Drop the footer parser (keep as fallback when `features` lacks `sources`). |
| Chat "Searching…" placeholder | Until the first token. | Until the sources event; then "Found N passages" / "Answering…". |
| Conversations list | On-device JSON. | `GET /conversations`, `GET /conversations/{id}`; write transcript as in the flow above. Keep the local store as cache. |
| Model picker | `GET /v1/models` ids only. | Show `sh_rag.parameter_size` / `quantization` from the same call. |
| Dashboard tiles and charts | Derived from the full document list. | `GET /stats` (`documents`, `chunks`, `by_type`, `indexes`, `workspace`). |
| Jobs card | Polls `/jobs` every 2 s; no cancel; `cancelled` falls through as success. | Subscribe to `GET /events`; add Cancel (`POST /jobs/{id}/cancel`); render `cancelled`; show `detail.completed/total` for pulls. |
| Pull a model | Free-text alert. | `GET /models/suggest?q=` as the user types. |
| Documents list | – | Row tap → `GET /documents/{key}` detail (passages, pages) with an "Open file" action. |
| Board | On-device cards; the server "has no task API". | `GET /board`, `POST /board/cards`, `PATCH`, `POST .../move` for drag and drop, `DELETE`. Subscribe to `event: board`. Column ids are `snake_case` (`in_progress`). |
| Settings → OpenWebUI link | Hard-codes port 3000. | Use `status.webui_url`. |
| Settings → connection test | Gateway `/healthz` + `/v1/models`, ingest `/status`. | Also read `features` and `auth_required`; prompt for the ingest key when required (store in Keychain, send as Bearer). |
| Connecting from a device | Edit docker-compose. | Set `BIND_ADDRESS=0.0.0.0`, `GATEWAY_API_KEY`, `INGEST_API_KEY` in `.env`. |

---

## Versioning

- `api.version` (ingest) and the gateway `version` identify this contract. Within version 2,
  changes are additive only: new fields, new endpoints, new `features` entries.
- Removing or renaming anything above means version 3 and a new `features` set.
- The ingest service's full OpenAPI schema is at `GET /openapi.json`.
