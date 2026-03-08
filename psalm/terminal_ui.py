from __future__ import annotations

TERMINAL_WIDTH = 80
INNER_WIDTH = TERMINAL_WIDTH - 2


def ellipsize(text: object, width: int) -> str:
    value = str(text)
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return "." * width
    return value[: width - 3] + "..."


def center_text(text: object, width: int = TERMINAL_WIDTH) -> str:
    return f"{ellipsize(text, width):^{width}}"


def frame_top(width: int = TERMINAL_WIDTH) -> str:
    return "┌" + "─" * (width - 2) + "┐"


def frame_bottom(width: int = TERMINAL_WIDTH) -> str:
    return "└" + "─" * (width - 2) + "┘"


def frame_line(text: object = "", width: int = TERMINAL_WIDTH) -> str:
    inner_width = width - 2
    return "│" + f"{ellipsize(text, inner_width):^{inner_width}}" + "│"


def divider(width: int = TERMINAL_WIDTH) -> str:
    return "─" * width


def section_header(title: str) -> str:
    return f"▸ {title}"


def kv_line(label: str, value: object, label_width: int = 12, width: int = TERMINAL_WIDTH) -> str:
    value_width = max(0, width - label_width - 1)
    return f"{label:<{label_width}} {ellipsize(value, value_width)}"


def result_top(title: str, width: int = TERMINAL_WIDTH) -> str:
    text = f"─ {title} "
    fill = max(0, width - 2 - len(text))
    return "┌" + text + ("─" * fill) + "┐"


def result_bottom(width: int = TERMINAL_WIDTH) -> str:
    return frame_bottom(width)
