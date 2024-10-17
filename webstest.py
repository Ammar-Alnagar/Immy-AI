import asyncio
import websockets

async def test_connection():
    try:
        async with websockets.connect("https://llm.criticalfutureglobal.com/api/v1/prediction/c9b49588-6fb9-493e-a86a-028964b307df") as websocket:
            print("Connected to the server!")
    except Exception as e:
        print(f"Failed to connect: {str(e)}")

asyncio.run(test_connection())
