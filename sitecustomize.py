import os
import sys


def _normalize_port_arg(raw_value: str) -> str | None:
    value = str(raw_value).strip().strip("\"'")

    if value.isdigit():
        return value

    # Handle common misconfigurations where "$PORT" is passed literally.
    if "PORT" in value or value.startswith("$"):
        env_port = os.getenv("PORT", "")
        if str(env_port).isdigit():
            return str(env_port)
        return "8000"

    return None


def _patch_cli_args() -> None:
    args = sys.argv

    for idx, arg in enumerate(args):
        if arg == "--port" and idx + 1 < len(args):
            normalized = _normalize_port_arg(args[idx + 1])
            if normalized is not None:
                args[idx + 1] = normalized
        elif arg.startswith("--port="):
            _, raw = arg.split("=", 1)
            normalized = _normalize_port_arg(raw)
            if normalized is not None:
                args[idx] = f"--port={normalized}"


_patch_cli_args()
