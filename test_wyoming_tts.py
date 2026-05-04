import asyncio
import json
import sys

from wyoming.client import AsyncTcpClient
from wyoming.event import Event


HOST = "127.0.0.1"
PORT = 10200


async def main():
    print("🧪 Connecting to Wyoming server...")

    # ✅ CORRECT WAY (Wyoming 1.8+)
    async with AsyncTcpClient(HOST, PORT) as client:

        print("📡 Sending: describe")

        await client.write_event(Event(type="describe"))

        print("\n📥 Waiting for response...\n")

        while True:
            event = await client.read_event()

            if event is None:
                print("❌ Connection closed")
                break

            print("=" * 60)
            print("EVENT:", event.type)

            if event.data:
                try:
                    print(json.dumps(event.data, indent=2))
                except Exception:
                    print(event.data)

            # Stop when we get full capability payload
            if event.type == "info":
                print("\n✅ INFO received")
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
