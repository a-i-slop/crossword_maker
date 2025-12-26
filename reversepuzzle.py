#!/usr/bin/env python3
"""
ipuz_to_tsv.py

Convert a crossword .ipuz file into a headerless .tsv:
    answer<TAB>clue

Additionally, if a clue ends with an enumeration like "(2, 5)" or "(3, 5, 2)",
the tool inserts spaces into the answer accordingly:
    (2, 5)      => space after 2nd letter
    (3, 5, 2)   => spaces after 3rd and 8th letters (3, 3+5)

Usage:
  python ipuz_to_tsv.py puzzle.ipuz output.tsv
  python ipuz_to_tsv.py puzzle.ipuz -o -              # write to stdout
  python ipuz_to_tsv.py puzzle.ipuz output.tsv --strip-enum
  python ipuz_to_tsv.py puzzle.ipuz output.tsv --skip-missing

Notes:
- Works with .ipuz files that are either pure JSON or wrapped like: ipuz({...})
- Answers are read from:
  1) clue entries (if they include an answer field / third element), else
  2) the 'solution' grid (Across/Down extraction)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Match enumeration at the very end with 2+ numbers separated by comma, dash, or slash:
#   "(2, 5)" or "(3,5,2)" or "(2-5)" or "(3/5/2)"
ENUM_RE = re.compile(r"\((\s*\d+\s*(?:[,\-/]\s*\d+\s*)+)\)\s*$")


def _unwrap_cell(cell: Any) -> Any:
    """IPUZ cells can be scalar or dict-like with 'cell'/'value' etc."""
    if isinstance(cell, dict):
        for k in ("cell", "value", "number", "contents", "data"):
            if k in cell:
                return cell[k]
        return None
    return cell


def _cell_number(puzzle_cell: Any) -> Optional[int]:
    """Extract the clue number if present in the puzzle grid cell."""
    v = _unwrap_cell(puzzle_cell)
    if isinstance(v, int) and v > 0:
        return v
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*", v)
        if m:
            return int(m.group(1))
    return None


def _is_block(puzzle_cell: Any, solution_cell: Any = None) -> bool:
    """Detect block squares."""
    pv = _unwrap_cell(puzzle_cell)
    if isinstance(pv, str) and pv.strip() == "#":
        return True
    if pv is None:
        # Some files may use null for blocks, but null can also mean unknown.
        # If solution explicitly marks '#', treat as block.
        sv = _unwrap_cell(solution_cell)
        if isinstance(sv, str) and sv.strip() == "#":
            return True

    sv = _unwrap_cell(solution_cell)
    if isinstance(sv, str) and sv.strip() == "#":
        return True

    return False


def _cell_solution(solution_cell: Any) -> str:
    """Get the solution string for a cell (could be multi-letter like 'QU')."""
    sv = _unwrap_cell(solution_cell)
    if sv is None:
        return ""
    if isinstance(sv, str):
        if sv.strip() == "#":
            return ""
        return sv
    if isinstance(sv, int):
        return str(sv)
    return str(sv)


def load_ipuz(path: Path) -> Dict[str, Any]:
    """Load .ipuz content (supports ipuz(<json>) wrapper)."""
    text = path.read_text(encoding="utf-8-sig").strip()

    # Typical wrapper: ipuz({...})
    if text.startswith("ipuz(") and text.endswith(")"):
        text = text[len("ipuz(") : -1].strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level IPUZ JSON must be an object/dict.")
        return data
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
            if not isinstance(data, dict):
                raise ValueError("Top-level IPUZ JSON must be an object/dict.")
            return data
        raise


def _get_grid(data: Dict[str, Any], *keys: str) -> Optional[List[List[Any]]]:
    for k in keys:
        v = data.get(k)
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v  # type: ignore[return-value]
    return None


def _start_across(block: List[List[bool]], y: int, x: int) -> bool:
    if block[y][x]:
        return False
    if x > 0 and not block[y][x - 1]:
        return False
    w = len(block[0])
    return (x + 1 < w) and (not block[y][x + 1])


def _start_down(block: List[List[bool]], y: int, x: int) -> bool:
    if block[y][x]:
        return False
    if y > 0 and not block[y - 1][x]:
        return False
    h = len(block)
    return (y + 1 < h) and (not block[y + 1][x])


def build_number_grid(puzzle: List[List[Any]], solution: Optional[List[List[Any]]]) -> Tuple[List[List[Optional[int]]], List[List[bool]]]:
    """Build numbering grid; uses existing numbers and fills missing start cells."""
    h = len(puzzle)
    w = len(puzzle[0]) if h else 0

    block = [
        [_is_block(puzzle[y][x], solution[y][x] if solution else None) for x in range(w)]
        for y in range(h)
    ]

    number = [[_cell_number(puzzle[y][x]) for x in range(w)] for y in range(h)]
    existing = {n for row in number for n in row if n is not None}

    next_num = 1
    for y in range(h):
        for x in range(w):
            if block[y][x]:
                continue
            if _start_across(block, y, x) or _start_down(block, y, x):
                if number[y][x] is None:
                    while next_num in existing:
                        next_num += 1
                    number[y][x] = next_num
                    existing.add(next_num)
                    next_num += 1
                else:
                    if number[y][x] >= next_num:
                        next_num = number[y][x] + 1

    return number, block


def extract_answers_from_solution(
    puzzle: List[List[Any]],
    solution: List[List[Any]],
) -> Dict[Tuple[str, int], str]:
    """Extract Across/Down answers keyed by (direction, number)."""
    number_grid, block = build_number_grid(puzzle, solution)
    h = len(puzzle)
    w = len(puzzle[0]) if h else 0

    answers: Dict[Tuple[str, int], str] = {}

    # Across
    for y in range(h):
        x = 0
        while x < w:
            if _start_across(block, y, x):
                num = number_grid[y][x]
                if num is not None:
                    letters: List[str] = []
                    xx = x
                    while xx < w and not block[y][xx]:
                        letters.append(_cell_solution(solution[y][xx]))
                        xx += 1
                    answers[("Across", num)] = "".join(letters).strip()
                # skip to end of word
                while x < w and not block[y][x]:
                    x += 1
            else:
                x += 1

    # Down
    for x in range(w):
        y = 0
        while y < h:
            if _start_down(block, y, x):
                num = number_grid[y][x]
                if num is not None:
                    letters = []
                    yy = y
                    while yy < h and not block[yy][x]:
                        letters.append(_cell_solution(solution[yy][x]))
                        yy += 1
                    answers[("Down", num)] = "".join(letters).strip()
                # skip to end of word
                while y < h and not block[y][x]:
                    y += 1
            else:
                y += 1

    return answers


def parse_clue_entry(entry: Any) -> Tuple[Optional[int], str, Optional[str]]:
    """
    Return (number, clue_text, answer_if_present)
    Handles common forms:
      [num, clue]
      [num, clue, answer]
      {"number": num, "clue": "...", "answer": "..."}
    """
    num: Any = None
    clue: Any = ""
    ans: Any = None

    if isinstance(entry, (list, tuple)):
        if len(entry) >= 1:
            num = entry[0]
        if len(entry) >= 2:
            clue = entry[1]
        if len(entry) >= 3:
            ans = entry[2]
    elif isinstance(entry, dict):
        num = entry.get("number", entry.get("num", entry.get("label", entry.get("id"))))
        clue = entry.get("clue", entry.get("hint", entry.get("text", "")))
        ans = entry.get("answer", entry.get("solution", entry.get("entry")))
    else:
        raise TypeError(f"Unsupported clue entry type: {type(entry).__name__}")

    n: Optional[int]
    if isinstance(num, int):
        n = num
    elif isinstance(num, str):
        m = re.search(r"\d+", num)
        n = int(m.group(0)) if m else None
    elif num is None:
        n = None
    else:
        try:
            n = int(num)
        except Exception:
            n = None

    clue_text = "" if clue is None else str(clue)
    ans_text = None if ans is None else str(ans)

    return n, clue_text, ans_text


def sanitize_hint(hint: str) -> str:
    """Make TSV-safe: collapse whitespace, remove tabs/newlines."""
    hint = hint.replace("\t", " ")
    hint = re.sub(r"\s+", " ", hint)
    return hint.strip()


def enum_parts_from_hint(hint: str) -> Optional[List[int]]:
    m = ENUM_RE.search(hint)
    if not m:
        return None
    parts = [int(x) for x in re.findall(r"\d+", m.group(1))]
    if len(parts) < 2:
        return None
    return parts


def insert_spaces_from_parts(answer: str, parts: Sequence[int]) -> str:
    """
    Insert spaces into answer after cumulative segment sizes (excluding last segment).
    Removes existing spaces/hyphens first to avoid double spacing.
    """
    raw = re.sub(r"[\s-]+", "", answer).strip()
    if not raw:
        return raw

    cut_positions: List[int] = []
    cum = 0
    for seg in parts[:-1]:
        cum += seg
        cut_positions.append(cum)

    out: List[str] = []
    for i, ch in enumerate(raw, start=1):
        out.append(ch)
        if i in cut_positions:
            out.append(" ")

    return "".join(out).strip()


def maybe_apply_enumeration_spacing(answer: str, hint: str) -> str:
    parts = enum_parts_from_hint(hint)
    if not parts:
        return answer
    return insert_spaces_from_parts(answer, parts)


def maybe_strip_enumeration(hint: str) -> str:
    return ENUM_RE.sub("", hint).rstrip()


def iter_clues_in_order(data: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    """
    Yield (direction, entry) in a reasonable order: Across first, then Down, then others.
    """
    clues = data.get("clues")
    if not isinstance(clues, dict):
        return

    keys = list(clues.keys())

    def key_rank(k: str) -> Tuple[int, int]:
        lk = k.lower()
        if "across" in lk:
            return (0, keys.index(k))
        if "down" in lk:
            return (1, keys.index(k))
        return (2, keys.index(k))

    for heading in sorted(keys, key=key_rank):
        entries = clues.get(heading)
        if not isinstance(entries, list):
            continue
        lk = heading.lower()
        if "across" in lk:
            direction = "Across"
        elif "down" in lk:
            direction = "Down"
        else:
            direction = heading  # unknown/custom
        for entry in entries:
            yield direction, entry


def convert_ipuz_to_tsv(
    ipuz_path: Path,
    out_path: Optional[Path],
    *,
    strip_enum: bool,
    skip_missing: bool,
) -> int:
    data = load_ipuz(ipuz_path)

    puzzle = _get_grid(data, "puzzle", "grid")
    if puzzle is None:
        print("ERROR: Could not find a puzzle grid under keys: 'puzzle' or 'grid'.", file=sys.stderr)
        return 2

    solution = _get_grid(data, "solution", "solutions", "answer", "answers")

    answers_from_grid: Dict[Tuple[str, int], str] = {}
    if solution is not None:
        try:
            answers_from_grid = extract_answers_from_solution(puzzle, solution)
        except Exception as e:
            print(f"WARNING: Failed to extract answers from solution grid: {e}", file=sys.stderr)

    # Choose output stream
    if out_path is None or str(out_path) == "-":
        out_f = sys.stdout
        close_out = False
    else:
        out_f = out_path.open("w", encoding="utf-8", newline="\n")
        close_out = True

    missing_count = 0
    written_count = 0

    try:
        for direction, raw_entry in iter_clues_in_order(data):
            num, hint, embedded_answer = parse_clue_entry(raw_entry)
            hint = sanitize_hint(hint)

            if strip_enum:
                hint_out = maybe_strip_enumeration(hint)
            else:
                hint_out = hint

            answer: Optional[str] = None

            # 1) Prefer embedded answer if present
            if embedded_answer is not None and embedded_answer.strip():
                answer = embedded_answer.strip()
            # 2) Else map from solution-grid extraction (only for standard directions)
            elif num is not None and direction in ("Across", "Down"):
                answer = answers_from_grid.get((direction, num))

            if answer is None or not answer.strip():
                missing_count += 1
                msg = f"Missing answer for {direction} {num if num is not None else '?'}: {hint}"
                if skip_missing:
                    print(f"WARNING: {msg}", file=sys.stderr)
                    continue
                else:
                    print(f"ERROR: {msg}", file=sys.stderr)
                    return 3

            answer = answer.strip()
            answer = maybe_apply_enumeration_spacing(answer, hint)

            # TSV row: answer<TAB>hint
            out_f.write(f"{answer}\t{hint_out}\n")
            written_count += 1

    finally:
        if close_out:
            out_f.close()

    if missing_count and skip_missing:
        print(f"Done with warnings: {missing_count} clue(s) missing answers; wrote {written_count} row(s).", file=sys.stderr)
    else:
        print(f"Done: wrote {written_count} row(s).", file=sys.stderr)

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert .ipuz -> headerless .tsv (answer<TAB>clue).")
    p.add_argument("ipuz", type=Path, help="Input .ipuz file")
    p.add_argument("tsv", nargs="?", type=Path, default=None, help="Output .tsv file (default: stdout)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output path (use '-' for stdout)")
    p.add_argument("--skip-missing", action="store_true", help="Skip clues that have no recoverable answer instead of failing.")

    args = p.parse_args(argv)

    out_path = args.output if args.output is not None else args.tsv
    if out_path is None:
        out_path = Path("-")

    return convert_ipuz_to_tsv(
        args.ipuz,
        out_path,
        strip_enum=True,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
