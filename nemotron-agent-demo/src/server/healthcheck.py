from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import requests


def check(host: str = "localhost", port: int = 8000, timeout: int = 30) -> bool:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Poll vLLM server for readiness")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)

    ok = check(args.host, args.port, args.timeout)
    if ok:
        print("healthcheck: server is ready")
        return 0
    print("healthcheck: server did not respond in time", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
