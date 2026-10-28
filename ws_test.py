"""ws_test.py — proves the WebSocket works AND survives a client dropping mid-conversation."""
import asyncio, json
import websockets
import httpx

URI = "ws://localhost:8000/ws/chat/ws_test"

async def main():
    # Client 1: normal multi-turn exchange
    async with websockets.connect(URI) as ws:
        await ws.send("What is the SLA for Priority 1 tickets?")
        r1 = json.loads(await ws.recv())
        print("Turn 1 answer :", r1["answer"][:70])

        await ws.send("Check ticket OLA-0006")
        r2 = json.loads(await ws.recv())
        print("Turn 2 ticket :", r2["ticket_id"])

        # Send a third message, then drop WITHOUT waiting for the reply
        await ws.send("What about Priority 2?")
    print("Client 1 disconnected mid-conversation")

    # Give the server a moment, then prove it is still alive
    await asyncio.sleep(3)
    health = httpx.get("http://localhost:8000/health").json()
    print("Server health after disconnect:", health)

    # Client 2: a fresh client still works
    async with websockets.connect("ws://localhost:8000/ws/chat/ws_test_2") as ws2:
        await ws2.send("How does Ola handle VIP customers?")
        r3 = json.loads(await ws2.recv())
        print("Client 2 answer:", r3["answer"][:70])

asyncio.run(main())