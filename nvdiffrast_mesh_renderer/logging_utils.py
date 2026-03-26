from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import pathlib


@dataclass(frozen=True)
class RunLogger:
    path: pathlib.Path
    echo: bool = False
    prefix: str | None = None

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def child(self, prefix: str) -> "RunLogger":
        if self.prefix:
            return RunLogger(path=self.path, echo=self.echo, prefix=f"{self.prefix} {prefix}")
        return RunLogger(path=self.path, echo=self.echo, prefix=prefix)

    def log(self, message: str, *, console: str = "echo", console_message: str | None = None) -> None:
        text = message if message.endswith("\n") else f"{message}\n"
        prefix = "" if self.prefix is None else f"{self.prefix} "
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = "".join(f"[{timestamp}] {prefix}{line}\n" for line in text.rstrip("\n").splitlines())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
        should_print = console == "always" or (console == "echo" and self.echo)
        if should_print:
            rendered = text if console_message is None else (console_message if console_message.endswith("\n") else f"{console_message}\n")
            print(rendered, end="")


def format_duration_ms(duration_ms: float | None) -> str:
    if duration_ms is None or not math.isfinite(duration_ms) or duration_ms < 0.0:
        return "0s"
    if duration_ms < 1000.0:
        return f"{duration_ms:.0f}ms"
    total_seconds = duration_ms / 1000.0
    if total_seconds < 60.0:
        return f"{total_seconds:.1f}s"
    total_minutes = int(total_seconds // 60.0)
    seconds = int(round(total_seconds % 60.0))
    if total_minutes < 60:
        return f"{total_minutes}m {seconds:02d}s"
    total_hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{total_hours}h {minutes:02d}m"


def estimate_remaining_ms(completed: int, total: int, elapsed_ms: float) -> float | None:
    if completed <= 0 or total <= completed or elapsed_ms <= 0.0:
        return None
    average_ms = elapsed_ms / float(completed)
    return average_ms * float(total - completed)


def format_path_notice(level: str, message: str, path: pathlib.Path) -> str:
    return f"[{level}] {message}\n{path}"
