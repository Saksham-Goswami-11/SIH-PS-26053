"""
Interactive CLI test client for KAVACH-2.5D WebSocket stream.
Connects to ws://localhost:8000/ws/stream_map and displays live telemetry and threats.
"""

import asyncio
import json
import websockets
import sys

URI = "ws://127.0.0.1:8000/ws/stream_map"


async def listen():
    print(f"\n📡 Connecting to KAVACH-2.5D stream at {URI}...")
    try:
        async with websockets.connect(URI, max_size=16 * 1024 * 1024) as ws:
            print("✅ Connected! Streaming live frames (Ctrl+C to stop)...\n")
            print(f"{'FRAME':<8} | {'ENGINE':<14} | {'FPS':<6} | {'LATENCY':<10} | {'RAW PTS':<9} | {'CELLS':<8} | {'SAVED %':<8} | {'THREATS'}")
            print("-" * 90)

            frame_count = 0
            while True:
                data = await ws.recv()
                payload = json.loads(data)

                header = payload.get("header", {})
                telemetry = payload.get("telemetry", {})
                threats = payload.get("threats", [])
                grid_data = payload.get("grid_data", [])

                fid = header.get("frame_id", 0)
                engine = header.get("active_engine", "UNKNOWN")
                fps = telemetry.get("fps", 0.0)
                lat = telemetry.get("latency_ms", 0.0)
                raw_n = telemetry.get("raw_points_count", 0)
                cells_m = telemetry.get("compressed_cells_count", len(grid_data))
                saved_pct = telemetry.get("memory_saved_percent", 0.0)
                n_threats = len(threats)

                threat_str = f"⚠️ {n_threats} alerts" if n_threats > 0 else "0"
                if n_threats > 0 and frame_count % 10 == 0:
                    sample = threats[0]
                    threat_str += f" (Closest: {sample.get('distance_m', 0):.1f}m, depth: {sample.get('depth_m', 0):.2f}m)"

                print(f"#{fid:<7} | {engine:<14} | {fps:>5.1f}  | {lat:>7.2f} ms | {raw_n:>8}  | {cells_m:>7}  | {saved_pct:>6.1f}%  | {threat_str}")
                frame_count += 1

    except ConnectionRefusedError:
        print(f"\n❌ Could not connect to {URI}. Is the server running?")
        print("Start it with: uvicorn backend.api.main:app --port 8000\n")
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\nDisconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        sys.exit(0)
