# Assignment 3 — PUCIT CGPA Guider with MCP, Memory and Streaming

**MCP server URL:** `<PASTE YOUR DEPLOYED URL, e.g. https://xxxx.onrender.com/mcp>`
**API key (only if the host needs auth):** `<PASTE KEY or write "none">`
**GitHub repo (server code):** `<PASTE LINK>`

## Run locally
1. Server (terminal 1): `cd server`, `pip install -r requirements.txt`, `python server.py` (serves http://127.0.0.1:8000/mcp)
2. Agent (terminal 2): `cd agent`, `pip install -r requirements.txt`, copy `.env.example` to `.env`, fill `GROQ_API_KEY`, `MCP_URL` (and `FASTMCP_KEY` if needed), then `python main.py`
3. Test the server alone: `fastmcp list server.py`

## What was added
- **Task 1:** all 7 tools from Assignment 1 shared by `server/server.py` (plain functions in `tools.py`; return types are `float | str` because tools return an error string on bad input).
- **Task 2:** agent gets tools with `MCPAdapter(...).list_tools()` (`tools/list`); every call is `tools/call` over the network. Everything is async.
- **Task 3:** `AsyncSqliteSaver` (`memory.db`), thread_id = person's name. Commands: `/switch`, `/history`, `/forget`.
- **Task 4:** `astream` with `stream_mode=["updates","messages"]`: tokens print live, tool calls/results print from node updates.
- **Task 5:** only `MCP_URL` and `FASTMCP_KEY` in `.env` change.
- **Task 6:** `tool_error_guard` middleware turns any tool failure into a ToolMessage the model can read; `repair_dangling_tool_calls` fixes old chats that already contain an unanswered tool call.
