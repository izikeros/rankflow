"""RankFlow CLI entry point."""

from __future__ import annotations

import subprocess  # nosec B404 - used only to launch the bundled Streamlit app
import sys
from pathlib import Path


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(0)

    command = sys.argv[1]
    if command == "ui":
        _launch_ui()
    elif command in ("--help", "-h"):
        _print_help()
    elif command == "--version":
        from rankflow._version import __version__

        print(f"rankflow {__version__}")
    else:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)


def _launch_ui():
    """Launch the Streamlit web UI."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is required for the web UI.\n"
            "Install it with: pip install rankflow[ui]"
        )
        sys.exit(1)

    app_path = Path(__file__).parent / "ui" / "app.py"
    store_path = sys.argv[2] if len(sys.argv) > 2 else "./experiments"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--",
        store_path,
    ]
    # Fixed argv list (shell=False); only store_path is user-supplied and is
    # passed as a positional arg to the app, not interpreted by a shell.
    sys.exit(subprocess.call(cmd))  # nosec B603


def _print_help():
    print(
        "Usage: rankflow <command>\n"
        "\n"
        "Commands:\n"
        "  ui [path]    Launch the web UI (default store: ./experiments)\n"
        "  --version    Show version\n"
        "  --help       Show this help\n"
    )


if __name__ == "__main__":
    main()
