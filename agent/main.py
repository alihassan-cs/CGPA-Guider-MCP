"""
main.py — terminal chat with memory (per-person thread), streaming, and MCP tools.

Commands:  /switch  /history  /forget  quit
"""
import asyncio

import aiosqlite
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_agent, repair_dangling_tool_calls

DB_PATH = "memory.db"


async def ainput(prompt: str) -> str:
    # input() blocks; run it in a thread so the event loop stays free
    return (await asyncio.to_thread(input, prompt)).strip()


async def list_threads(memory) -> list[str]:
    names = set()
    async for cp in memory.alist(None):
        names.add(cp.config["configurable"]["thread_id"])
    names = sorted(names)
    print("Saved chats:", ", ".join(names) if names else "none yet")
    return names


async def show_history(app, config) -> None:
    state = await app.aget_state(config)
    msgs = state.values.get("messages", [])
    if not msgs:
        print("(no history)")
        return
    for m in msgs:
        if m.type == "human":
            print(f"You:   {m.text}")
        elif m.type == "ai":
            if m.text.strip():
                print(f"Agent: {m.text}")
            for tc in getattr(m, "tool_calls", None) or []:
                print(f"  [tool call] {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"  [tool result] {m.text}")


async def run_turn(app, text: str, config) -> None:
    """Stream one turn: tokens as they arrive + tool progress from node updates."""
    inputs = {"messages": [HumanMessage(text)]}
    print("Agent: ", end="", flush=True)
    async for chunk in app.astream(
        inputs, config, stream_mode=["updates", "messages"], version="v2"
    ):
        if chunk["type"] == "messages":
            token, meta = chunk["data"]
            if meta["langgraph_node"] == "model" and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            for node, update in chunk["data"].items():
                for m in (update or {}).get("messages", []):
                    if node == "model":
                        for tc in getattr(m, "tool_calls", None) or []:
                            print(f"\n  [calling {tc['name']} {tc['args']} ...]", flush=True)
                    elif node == "tools":
                        print(f"  [{m.name} -> {m.text}]", flush=True)
                        print("Agent: ", end="", flush=True)
    print("\n")


async def heal(app, config) -> None:
    """Add an answer for any unanswered tool call saved in this thread."""
    state = await app.aget_state(config)
    fixes = repair_dangling_tool_calls(state.values.get("messages", []))
    if fixes:
        await app.aupdate_state(config, {"messages": fixes})


async def main() -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        memory = AsyncSqliteSaver(conn)
        try:
            app = await build_agent(memory)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not reach the MCP server ({type(exc).__name__}: {exc}).")
            print("Start it (cd server; python server.py) or check MCP_URL in .env.")
            return

        print("PUCIT GPA Assistant — /switch /history /forget, 'quit' to exit.\n")
        await list_threads(memory)
        name = await ainput("Who is chatting? ")
        config = {"configurable": {"thread_id": name}}

        while True:
            try:
                text = await ainput(f"{name}> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            low = text.lower()
            if low in {"quit", "exit"}:
                break
            if low == "/switch":
                await list_threads(memory)
                name = await ainput("Who is chatting? ")
                config = {"configurable": {"thread_id": name}}
                continue
            if low == "/history":
                await show_history(app, config)
                continue
            if low == "/forget":
                await memory.adelete_thread(name)
                print(f"Deleted saved chat for {name}.")
                continue
            try:
                await heal(app, config)
                await run_turn(app, text, config)
            except Exception as exc:  # noqa: BLE001
                print(f"\nSomething went wrong ({type(exc).__name__}: {exc}). Try again.\n")


if __name__ == "__main__":
    asyncio.run(main())
