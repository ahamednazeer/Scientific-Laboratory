from __future__ import annotations

import argparse

import uvicorn

from app.core.config import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run Scientific Laboratory AR backend")
    parser.add_argument("--host", default=settings.host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.port, help="Bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.app_env.lower() == "development",
        help="Enable auto-reload (default: enabled in development)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload even in development",
    )
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reload_enabled = args.reload and not args.no_reload
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
