from __future__ import annotations

import argparse
import csv
import collections
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]  # (x, y)

H_BIT = 1
V_BIT = 2


@dataclass(frozen=True)
class Placement:
    """One placed word in the grid."""
    word: str
    x: int
    y: int
    direction: str  # 'H' or 'V'


@dataclass(frozen=True)
class PlaceEval:
    """Result of checking feasibility + geometry in one pass."""
    intersections: int
    new_letters: int
    touches: int
    minx: int
    maxx: int
    miny: int
    maxy: int


@dataclass(frozen=True)
class WordMeta:
    word: str
    length: int
    unique_letters: Tuple[str, ...]
    # letter -> indices where it occurs in the word (for dedup)
    letter_to_indices: Dict[str, Tuple[int, ...]]


def _build_word_meta(words: Sequence[str]) -> Dict[str, WordMeta]:
    meta: Dict[str, WordMeta] = {}
    for w in words:
        idxs: Dict[str, List[int]] = {}
        for i, ch in enumerate(w):
            idxs.setdefault(ch, []).append(i)
        meta[w] = WordMeta(
            word=w,
            length=len(w),
            unique_letters=tuple(sorted(idxs.keys())),
            letter_to_indices={ch: tuple(v) for ch, v in idxs.items()},
        )
    return meta


@dataclass
class CrosswordLayout:
    """
    A sparse crossword-like word arrangement.

    Grid is stored as a dict from (x, y) -> letter.
    Any coordinate not in `grid` is treated as a black cell when exporting.
    """
    clearance: int = 2

    grid: Dict[Coord, str] = field(default_factory=dict)
    # Bitmask per occupied cell: H_BIT and/or V_BIT
    dir_mask: Dict[Coord, int] = field(default_factory=dict)
    # Owner placement index for each direction
    owner_h: Dict[Coord, int] = field(default_factory=dict)
    owner_v: Dict[Coord, int] = field(default_factory=dict)

    # Crossable anchors: for placing direction D, you can intersect at coords where ONLY the other direction exists.
    # crossable['H'][ch] are coords containing ch with ONLY V present. (i.e., H can cross there)
    # crossable['V'][ch] are coords containing ch with ONLY H present. (i.e., V can cross there)
    crossable: Dict[str, DefaultDict[str, Set[Coord]]] = field(
        default_factory=lambda: {
            "H": collections.defaultdict(set),
            "V": collections.defaultdict(set),
        }
    )

    placements: List[Placement] = field(default_factory=list)

    # Number of disconnected placements (placed with 0 intersections into an existing non-empty grid)
    islands: int = 0

    # Bounding box for occupied letters
    minx: int = 0
    maxx: int = -1
    miny: int = 0
    maxy: int = -1

    def clone(self) -> "CrosswordLayout":
        """Deep-ish clone sufficient for beam search branching."""
        new = CrosswordLayout(clearance=self.clearance)
        new.grid = self.grid.copy()
        new.dir_mask = self.dir_mask.copy()
        new.owner_h = self.owner_h.copy()
        new.owner_v = self.owner_v.copy()

        new.crossable = {
            "H": collections.defaultdict(set),
            "V": collections.defaultdict(set),
        }
        for d in ("H", "V"):
            for ch, s in self.crossable[d].items():
                if s:
                    new.crossable[d][ch] = set(s)

        new.placements = list(self.placements)
        new.islands = self.islands
        new.minx, new.maxx, new.miny, new.maxy = self.minx, self.maxx, self.miny, self.maxy
        return new

    def bbox(self) -> Tuple[int, int, int, int]:
        if not self.grid:
            return (0, -1, 0, -1)
        return (self.minx, self.maxx, self.miny, self.maxy)

    def width(self) -> int:
        return 0 if not self.grid else self.maxx - self.minx + 1

    def height(self) -> int:
        return 0 if not self.grid else self.maxy - self.miny + 1

    def area(self) -> int:
        return self.width() * self.height()

    def density(self) -> float:
        return 0.0 if not self.grid else len(self.grid) / max(1, self.area())

    def centroid(self) -> Tuple[float, float]:
        """Approximate center of mass using bbox center (fast)."""
        if not self.grid:
            return (0.0, 0.0)
        return ((self.minx + self.maxx) / 2.0, (self.miny + self.maxy) / 2.0)
    
    def holes_in_bbox(self) -> int:
        """Empty cells inside current bbox."""
        if not self.grid:
            return 0
        return max(0, self.area() - len(self.grid))

    def perimeter(self) -> int:
        """Exposed edge count of occupied cells (lower is more 'blob-like')."""
        if not self.grid:
            return 0
        per = 0
        for (x, y) in self.grid.keys():
            if (x - 1, y) not in self.grid:
                per += 1
            if (x + 1, y) not in self.grid:
                per += 1
            if (x, y - 1) not in self.grid:
                per += 1
            if (x, y + 1) not in self.grid:
                per += 1
        return per

    def degree1_cells(self) -> int:
        """Count occupied cells with exactly one orthogonal occupied neighbor."""
        if not self.grid:
            return 0
        d1 = 0
        for (x, y) in self.grid.keys():
            deg = 0
            if (x - 1, y) in self.grid:
                deg += 1
            if (x + 1, y) in self.grid:
                deg += 1
            if (x, y - 1) in self.grid:
                deg += 1
            if (x, y + 1) in self.grid:
                deg += 1
            if deg == 1:
                d1 += 1
        return d1

    def _word_bbox(self, word_len: int, x: int, y: int, direction: str) -> Tuple[int, int, int, int]:
        if direction == "H":
            return x, x + word_len - 1, y, y
        else:
            return x, x, y, y + word_len - 1

    def _recompute_bbox(self) -> None:
        if not self.grid:
            self.minx, self.maxx, self.miny, self.maxy = 0, -1, 0, -1
            return
        xs = [p[0] for p in self.grid.keys()]
        ys = [p[1] for p in self.grid.keys()]
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)

    def _perp_word_after_letter(
        self,
        x: int,
        y: int,
        letter: str,
        placed_direction: str,
    ) -> Optional[str]:
        """
        If we were to place `letter` at (x, y) as part of a word placed in `placed_direction`,
        return the perpendicular entry (Across/Down) that would exist through (x, y).
        Returns None if the perpendicular run length would be < 2.
        """
        if placed_direction == "H":
            # Perpendicular is vertical (Down).
            y0 = y
            while (x, y0 - 1) in self.grid:
                y0 -= 1
            y1 = y
            while (x, y1 + 1) in self.grid:
                y1 += 1
            if (y1 - y0 + 1) < 2:
                return None
            chars: List[str] = []
            for yy in range(y0, y1 + 1):
                chars.append(letter if yy == y else self.grid[(x, yy)])
            return "".join(chars)

        if placed_direction == "V":
            # Perpendicular is horizontal (Across).
            x0 = x
            while (x0 - 1, y) in self.grid:
                x0 -= 1
            x1 = x
            while (x1 + 1, y) in self.grid:
                x1 += 1
            if (x1 - x0 + 1) < 2:
                return None
            chars: List[str] = []
            for xx in range(x0, x1 + 1):
                chars.append(letter if xx == x else self.grid[(xx, y)])
            return "".join(chars)

        raise ValueError("placed_direction must be 'H' or 'V'")

    def eval_place(
        self,
        word: str,
        x: int,
        y: int,
        direction: str,
        *,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        require_intersection: bool = True,
        valid_words: Optional[Set[str]] = None,
    ) -> Optional[PlaceEval]:
        """
        Check if a word can be placed, computing geometry in one scan.
        Returns PlaceEval if valid, else None.
        """
        L = len(word)
        if L == 0:
            return None

        if direction == "H":
            # Endcaps must be empty
            if (x - 1, y) in self.grid or (x + L, y) in self.grid:
                return None
            dx, dy = 1, 0
            d_bit = H_BIT
            # perpendicular offsets for clearance
            perp = ((0, -1), (0, 1))
        elif direction == "V":
            if (x, y - 1) in self.grid or (x, y + L) in self.grid:
                return None
            dx, dy = 0, 1
            d_bit = V_BIT
            perp = ((-1, 0), (1, 0))
        else:
            raise ValueError("direction must be 'H' or 'V'")

        # Candidate bbox
        wminx, wmaxx, wminy, wmaxy = self._word_bbox(L, x, y, direction)
        if not self.grid:
            minx, maxx, miny, maxy = wminx, wmaxx, wminy, wmaxy
        else:
            minx = min(self.minx, wminx)
            maxx = max(self.maxx, wmaxx)
            miny = min(self.miny, wminy)
            maxy = max(self.maxy, wmaxy)

        # Global size constraints (O(1))
        if max_width is not None and (maxx - minx + 1) > max_width:
            return None
        if max_height is not None and (maxy - miny + 1) > max_height:
            return None

        intersections = 0
        new_letters = 0
        touches = 0

        cx, cy = x, y
        for i in range(L):
            ch = word[i]
            pos = (cx, cy)
            existing = self.grid.get(pos)
            if existing is not None:
                if existing != ch:
                    return None
                mask = self.dir_mask.get(pos, 0)
                if mask & d_bit:
                    return None  # already occupied in this direction
                intersections += 1
            else:
                # New cell: clearance check perpendicular
                new_letters += 1
                if self.clearance > 0:
                    for dist in range(1, self.clearance + 1):
                        for ox, oy in perp:
                            if (cx + ox * dist, cy + oy * dist) in self.grid:
                                return None
                else:
                    # clearance == 0: touching is only allowed if it doesn't create invalid
                    # perpendicular (Across/Down) entries. Otherwise you get stray 2-letter "words".
                    cell_touches = 0
                    for ox, oy in perp:
                        if (cx + ox, cy + oy) in self.grid:
                            cell_touches += 1
                    if cell_touches:
                        touches += cell_touches
                        if valid_words is not None:
                            perp_word = self._perp_word_after_letter(cx, cy, ch, direction)
                            if perp_word is not None and perp_word not in valid_words:
                                return None
            
            cx += dx
            cy += dy

        if require_intersection and self.grid and intersections == 0:
            return None

        return PlaceEval(
            intersections=intersections,
            new_letters=new_letters,
            touches=touches,
            minx=minx,
            maxx=maxx,
            miny=miny,
            maxy=maxy,
        )

    def add_word(self, word: str, x: int, y: int, direction: str, *, intersections: Optional[int] = None) -> None:
        """
        Mutates the layout by placing the word. Caller should have validated placement via eval_place.
        `intersections` is used to update island count without rescanning.
        """
        L = len(word)
        prior_nonempty = bool(self.grid)

        if direction == "H":
            dx, dy = 1, 0
            d_bit = H_BIT
            opp_dir = "V"
        elif direction == "V":
            dx, dy = 0, 1
            d_bit = V_BIT
            opp_dir = "H"
        else:
            raise ValueError("direction must be 'H' or 'V'")

        placement_index = len(self.placements)
        self.placements.append(Placement(word=word, x=x, y=y, direction=direction))

        # Update bbox
        wminx, wmaxx, wminy, wmaxy = self._word_bbox(L, x, y, direction)
        if self.maxx == -1:
            self.minx, self.maxx, self.miny, self.maxy = wminx, wmaxx, wminy, wmaxy
        else:
            self.minx = min(self.minx, wminx)
            self.maxx = max(self.maxx, wmaxx)
            self.miny = min(self.miny, wminy)
            self.maxy = max(self.maxy, wmaxy)

        cx, cy = x, y
        for i in range(L):
            ch = word[i]
            pos = (cx, cy)
            existed = pos in self.grid
            if not existed:
                self.grid[pos] = ch
                self.dir_mask[pos] = d_bit
                if direction == "H":
                    self.owner_h[pos] = placement_index
                else:
                    self.owner_v[pos] = placement_index
                # New single-direction cell is crossable in the opposite direction
                self.crossable[opp_dir][ch].add(pos)
            else:
                # Must be same letter; if not, it's a bug in caller usage.
                if self.grid[pos] != ch:
                    raise ValueError("add_word called with conflicting letter; call eval_place first")
                old_mask = self.dir_mask.get(pos, 0)
                if old_mask & d_bit:
                    raise ValueError("add_word called but direction already occupied; call eval_place first")
                self.dir_mask[pos] = old_mask | d_bit

                if direction == "H":
                    self.owner_h[pos] = placement_index
                else:
                    self.owner_v[pos] = placement_index

                # After we add the second direction, it becomes an intersection and is no longer crossable.
                self.crossable[direction][ch].discard(pos)

            cx += dx
            cy += dy

        if intersections is None:
            # Slow path: compute intersections if caller didn't provide
            ev = self.eval_place(word, x, y, direction, require_intersection=False)
            intersections = 0 if ev is None else ev.intersections

        if prior_nonempty and intersections == 0:
            self.islands += 1

    def connected_components(self) -> int:
        """Number of connected components of placed words (connected via intersections)."""
        n = len(self.placements)
        if n == 0:
            return 0
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for pos, mask in self.dir_mask.items():
            if mask == (H_BIT | V_BIT):
                a = self.owner_h.get(pos)
                b = self.owner_v.get(pos)
                if a is not None and b is not None:
                    union(a, b)

        return len({find(i) for i in range(n)})

    def save_ipuz(
        self,
        filename: str,
        title: str = "Sparse Crossword",
        clues_map: Optional[Dict[str, str]] = None,
        structure_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Save the layout to an .ipuz file (JSON format)."""
        if not self.grid:
            print("Grid empty, skipping ipuz save.")
            return

        clues_map = clues_map or {}
        structure_map = structure_map or {}

        minx, maxx, miny, maxy = self.bbox()
        w, h = self.width(), self.height()

        puzzle_grid: List[List[Any]] = [[None] * w for _ in range(h)]
        solution_grid: List[List[Any]] = [[None] * w for _ in range(h)]
        clues: Dict[str, List[List[Any]]] = {"Across": [], "Down": []}

        number = 1
        for r in range(h):
            for c in range(w):
                x = minx + c
                y = miny + r

                if (x, y) not in self.grid:
                    puzzle_grid[r][c] = "#"
                    solution_grid[r][c] = "#"
                    continue

                letter = self.grid[(x, y)]
                solution_grid[r][c] = letter

                left_blk = (x - 1, y) not in self.grid
                right_let = (x + 1, y) in self.grid
                top_blk = (x, y - 1) not in self.grid
                bot_let = (x, y + 1) in self.grid

                is_across = left_blk and right_let
                is_down = top_blk and bot_let

                cell_val = 0
                if is_across or is_down:
                    cell_val = number
                    number += 1

                    if is_across:
                        chars = []
                        cx = x
                        while (cx, y) in self.grid:
                            chars.append(self.grid[(cx, y)])
                            cx += 1
                        word_str = "".join(chars)
                        hint = clues_map.get(word_str, f"Clue for {word_str}")
                        orig = structure_map.get(word_str, "")
                        if " " in orig:
                            lengths = [len(part) for part in orig.split()]
                            hint = f"{hint} ({', '.join(map(str, lengths))})"
                        clues["Across"].append([cell_val, hint])

                    if is_down:
                        chars = []
                        cy = y
                        while (x, cy) in self.grid:
                            chars.append(self.grid[(x, cy)])
                            cy += 1
                        word_str = "".join(chars)
                        hint = clues_map.get(word_str, f"Clue for {word_str}")
                        orig = structure_map.get(word_str, "")
                        if " " in orig:
                            lengths = [len(part) for part in orig.split()]
                            hint = f"{hint} ({', '.join(map(str, lengths))})"
                        clues["Down"].append([cell_val, hint])

                puzzle_grid[r][c] = cell_val

        ipuz_data = {
            "version": "http://ipuz.org/v2",
            "kind": ["http://ipuz.org/crossword#1"],
            "dimensions": {"width": w, "height": h},
            "puzzle": puzzle_grid,
            "solution": solution_grid,
            "clues": clues,
            "title": title,
            "origin": "Generated by Python Sparse Crossword Generator",
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(ipuz_data, f, indent=2)
        print(f"Saved {filename}")


def _clean_word(w: str) -> str:
    return re.sub(r"[^A-Za-z]", "", w.strip()).upper()


def _sanitize_word_list(words: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for w_raw in words:
        w = _clean_word(w_raw)
        if len(w) < 2:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


@dataclass
class SearchState:
    layout: CrosswordLayout
    remaining: List[str]
    key: Tuple[int, int, float, int]


def _aspect_ratio(layout: CrosswordLayout) -> float:
    w, h = layout.width(), layout.height()
    if w == 0 or h == 0:
        return 1.0
    return max(w / h, h / w)  # 1.0 is perfect square


def _state_key(layout: CrosswordLayout, *, target_density: float) -> Tuple[int, int, float, float, int]:
    placed = len(layout.placements)
    density_diff = abs(layout.density() - target_density)
    ar = _aspect_ratio(layout)
    # Prefer smaller max side length over bbox area (area tends to reward rectangles)
    max_side = max(layout.width(), layout.height())
    holes = layout.holes_in_bbox()
    per = layout.perimeter()
    d1 = layout.degree1_cells()
    return (
        placed, 
        -holes,      # fewer holes
        -layout.islands, 
        -(ar - 1.0) * 10.0, 
        -density_diff, 
        -max_side,
        -per,        # less perimeter
        -d1,         # fewer dangling cells
    )


def _placement_score(
    layout: CrosswordLayout,
    meta: WordMeta,
    x: int,
    y: int,
    direction: str,
    ev: PlaceEval,
    *,
    target_density: float,
    progress: float,
    rng: random.Random,
    island: bool,
    touch_penalty: float,
) -> float:
    """
    Score a candidate placement. Higher is better.
    """
    new_w = ev.maxx - ev.minx + 1
    new_h = ev.maxy - ev.miny + 1
    new_area = new_w * new_h

    new_filled = len(layout.grid) + ev.new_letters
    new_density = new_filled / max(1, new_area)

    holes = max(0, new_area - new_filled)

    # centroid / gravity
    cx, cy = layout.centroid()
    half = (meta.length - 1) / 2.0
    if direction == "H":
        midx, midy = x + half, y
    else:
        midx, midy = x, y + half
    dist = math.hypot(midx - cx, midy - cy)

    # Strongly favor crossings; increasingly so later.
    w_inter = 250.0 + 400.0 * progress
    w_inter2 = 60.0 + 80.0 * progress  # bonus for multi-cross

    # Annealed weights: care more about density/aspect later.
    w_density = 20.0 + 120.0 * progress
    w_aspect = 40.0 + 80.0 * progress

    # Anneal: early allow exploration, late punish holes hard.
    w_holes = 5.0 + 80.0 * progress

    score_inter = ev.intersections * w_inter + (ev.intersections ** 2) * w_inter2
    score_density = -abs(new_density - target_density) * w_density

    score_gravity = -dist * 1.5
    score_aspect = -((new_w - new_h) ** 2) * w_aspect

    score_holes = -holes * w_holes

    island_penalty = -500.0 if island else 0.0

    # Only meaningful when clearance==0, but safe either way.
    touch_pen = -touch_penalty * ev.touches

    # Tiny noise to break ties across attempts
    noise = rng.random() * 1e-6

    return score_inter + score_density + score_gravity + score_aspect + score_holes + island_penalty + touch_pen + noise


def _approx_anchor_potential(layout: CrosswordLayout, wm: WordMeta) -> int:
    """
    Fast heuristic: sum of crossable anchor counts for letters in this word.
    Only counts *possible* intersection anchors (correct direction availability).
    """
    total = 0
    chmap_h = layout.crossable["H"]
    chmap_v = layout.crossable["V"]
    for ch in wm.unique_letters:
        total += len(chmap_h.get(ch, ())) + len(chmap_v.get(ch, ()))
    return total


def _gen_intersecting_candidates(
    layout: CrosswordLayout,
    wm: WordMeta,
    *,
    max_width: Optional[int],
    max_height: Optional[int],
    target_density: float,
    progress: float,
    rng: random.Random,
    keep_top: int,
    touch_penalty: float,
    valid_words: Optional[Set[str]] = None,
) -> List[Tuple[float, int, int, str]]:
    """
    Generate top-N intersecting candidates for a word, deduplicated.
    Returns list of (score, x, y, direction).
    """
    import heapq

    heap: List[Tuple[float, int, int, str]] = []  # min-heap by score
    seen: Set[Tuple[int, int, str]] = set()

    for ch, idxs in wm.letter_to_indices.items():
        # direction 'H' means placing horizontally, so it can only cross cells with ONLY V.
        for direction in ("H", "V"):
            anchors = layout.crossable[direction].get(ch)
            if not anchors:
                continue

            for gx, gy in anchors:
                for i in idxs:
                    if direction == "H":
                        cx, cy = gx - i, gy
                    else:
                        cx, cy = gx, gy - i

                    key = (cx, cy, direction)
                    if key in seen:
                        continue
                    seen.add(key)

                    ev = layout.eval_place(
                        wm.word,
                        cx,
                        cy,
                        direction,
                        max_width=max_width,
                        max_height=max_height,
                        require_intersection=True,
                        valid_words=valid_words,
                    )
                    if ev is None:
                        continue

                    score = _placement_score(
                        layout,
                        wm,
                        cx,
                        cy,
                        direction,
                        ev,
                        target_density=target_density,
                        progress=progress,
                        rng=rng,
                        island=False,
                        touch_penalty=touch_penalty,
                    )

                    if len(heap) < keep_top:
                        heapq.heappush(heap, (score, cx, cy, direction))
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, (score, cx, cy, direction))

    heap.sort(reverse=True, key=lambda t: t[0])
    return heap


def _gen_island_candidates(
    layout: CrosswordLayout,
    wm: WordMeta,
    *,
    max_width: Optional[int],
    max_height: Optional[int],
    target_density: float,
    progress: float,
    rng: random.Random,
    keep_top: int,
    island_gap: int,
    touch_penalty: float,
    valid_words: Optional[Set[str]] = None,
) -> List[Tuple[float, int, int, str]]:
    """
    Generate a small set of good island placements around the bbox.
    Returns list of (score, x, y, direction).
    """
    import heapq

    heap: List[Tuple[float, int, int, str]] = []

    if not layout.grid:
        ev = layout.eval_place(
            wm.word,
            0,
            0,
            "H",
            max_width=max_width,
            max_height=max_height,
            require_intersection=False,
            valid_words=valid_words,
        )
        if ev is None:
            return []
        score = _placement_score(
            layout,
            wm,
            0,
            0,
            "H",
            ev,
            target_density=target_density,
            progress=progress,
            rng=rng,
            island=False,
            touch_penalty=touch_penalty,
            valid_words=valid_words,
        )
        return [(score, 0, 0, "H")]

    minx, maxx, miny, maxy = layout.bbox()
    gap = island_gap + layout.clearance
    L = wm.length

    cx_box = (minx + maxx) // 2
    cy_box = (miny + maxy) // 2

    # Candidate starts: align to top/mid/bottom (or left/mid/right) around bbox edges.
    x_starts = [minx, maxx - L + 1, cx_box - L // 2]
    y_starts = [miny, maxy - L + 1, cy_box - L // 2]

    positions: List[Tuple[int, int, str]] = []

    # Horizontal above and below
    for y0 in (miny - gap, maxy + gap):
        for x0 in x_starts:
            positions.append((x0, y0, "H"))

    # Vertical left and right
    for x0 in (minx - gap, maxx + gap):
        for y0 in y_starts:
            positions.append((x0, y0, "V"))

    for x0, y0, d in positions:
        ev = layout.eval_place(
            wm.word,
            x0,
            y0,
            d,
            max_width=max_width,
            max_height=max_height,
            require_intersection=False,
            valid_words=valid_words,
        )
        if ev is None:
            continue

        island_flag = bool(layout.grid) and ev.intersections == 0
        score = _placement_score(
            layout,
            wm,
            x0,
            y0,
            d,
            ev,
            target_density=target_density,
            progress=progress,
            rng=rng,
            island=island_flag,
            touch_penalty=touch_penalty,
        )

        if len(heap) < keep_top:
            heapq.heappush(heap, (score, x0, y0, d))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, x0, y0, d))

    heap.sort(reverse=True, key=lambda t: t[0])
    return heap


def _beam_attempt(
    words: Sequence[str],
    meta: Dict[str, WordMeta],
    *,
    rng: random.Random,
    clearance: int,
    target_density: float,
    touch_penalty: float,
    max_width: Optional[int],
    max_height: Optional[int],
    allow_islands: bool,
    island_gap: int,
    beam_width: int,
    branch_factor: int,
    word_choices: int,
    max_word_scan: int,
    valid_words: Set[str],
) -> Tuple[CrosswordLayout, List[str]]:
    """
    One attempt using beam search (beam_width=1 behaves like an improved greedy).
    """
    if not words:
        return CrosswordLayout(clearance=clearance), []

    layout0 = CrosswordLayout(clearance=clearance)
    # Seed with the first word for stability
    first = words[0]
    layout0.add_word(first, 0, 0, "H", intersections=0)

    remaining0 = list(words[1:])
    total_words = len(words)

    initial = SearchState(layout=layout0, remaining=remaining0, key=_state_key(layout0, target_density=target_density))
    beam: List[SearchState] = [initial]

    # Expand up to placing all remaining words
    for _step in range(total_words - 1):
        next_states: List[SearchState] = []
        any_expanded = False

        for state in beam:
            if not state.remaining:
                next_states.append(state)
                continue

            layout = state.layout
            remaining = state.remaining

            # Progress factor for annealing
            progress = len(layout.placements) / max(1, total_words)

            # Build list of connectable words (approx) for MRV ordering
            connectable: List[Tuple[int, int, str]] = []
            for w in remaining:
                pot = _approx_anchor_potential(layout, meta[w])
                if pot > 0:
                    # sort key: fewest anchors first, then longer first
                    connectable.append((pot, -meta[w].length, w))

            # We'll gather the best children across a small scan of words.
            # Cap the number of children per state to keep branching under control.
            max_children = max(1, branch_factor * max(1, word_choices))
            import heapq
            child_heap: List[Tuple[float, int, int, str, str]] = []  # (score, x, y, dir, word)

            def push_child(score: float, x: int, y: int, d: str, w: str) -> None:
                if len(child_heap) < max_children:
                    heapq.heappush(child_heap, (score, x, y, d, w))
                else:
                    if score > child_heap[0][0]:
                        heapq.heapreplace(child_heap, (score, x, y, d, w))

            scanned_words = 0

            if connectable:
                connectable.sort(key=lambda t: (t[0], t[1]))  # pot asc, -len asc => longer first
                for _, __, w in connectable[:max_word_scan]:
                    scanned_words += 1
                    wm = meta[w]
                    cands = _gen_intersecting_candidates(
                        layout,
                        wm,
                        max_width=max_width,
                        max_height=max_height,
                        target_density=target_density,
                        progress=progress,
                        rng=rng,
                        keep_top=branch_factor,
                        touch_penalty=touch_penalty,
                        valid_words=valid_words,
                    )
                    for score, x, y, d in cands:
                        push_child(score, x, y, d, w)
                    # If we've already found enough children, stop scanning words early
                    if len(child_heap) >= max_children and scanned_words >= word_choices:
                        break

            # If we found no intersecting children, maybe do an island fallback
            if not child_heap and allow_islands:
                # Prefer long words for islands
                w = max(remaining, key=lambda ww: meta[ww].length)
                wm = meta[w]
                cands = _gen_island_candidates(
                    layout,
                    wm,
                    max_width=max_width,
                    max_height=max_height,
                    target_density=target_density,
                    progress=progress,
                    rng=rng,
                    keep_top=branch_factor,
                    island_gap=island_gap,
                    touch_penalty=touch_penalty,
                    valid_words=valid_words,
                )
                for score, x, y, d in cands:
                    push_child(score, x, y, d, w)

            if not child_heap:
                # Terminal: can't expand
                next_states.append(state)
                continue

            # Expand children
            child_heap.sort(reverse=True, key=lambda t: t[0])
            for score, x, y, d, w in child_heap:
                new_layout = layout.clone()
                ev = new_layout.eval_place(
                    w,
                    x,
                    y,
                    d,
                    max_width=max_width,
                    max_height=max_height,
                    require_intersection=(not allow_islands),
                    valid_words=valid_words,
                )
                if ev is None:
                    # Shouldn't happen, but skip if mismatch
                    continue
                new_layout.add_word(w, x, y, d, intersections=ev.intersections)

                new_remaining = remaining.copy()
                try:
                    new_remaining.remove(w)
                except ValueError:
                    continue

                key = _state_key(new_layout, target_density=target_density)
                next_states.append(SearchState(layout=new_layout, remaining=new_remaining, key=key))
                any_expanded = True

        if not any_expanded:
            break

        # Beam prune
        next_states.sort(key=lambda s: s.key, reverse=True)
        beam = next_states[: max(1, beam_width)]

    # Choose best terminal state from beam
    beam.sort(key=lambda s: s.key, reverse=True)
    best = beam[0]
    return best.layout, best.remaining


def generate_sparse_crossword(
    words: Sequence[str],
    *,
    attempts: int = 50,
    seed: Optional[int] = None,
    clearance: int = 2,
    touch_penalty: float = 0.0,
    target_density: float = 0.5,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    allow_islands: bool = False,
    island_gap: int = 3,
    beam_width: int = 1,
    branch_factor: int = 12,
    word_choices: int = 3,
    max_word_scan: int = 12,
) -> Tuple[CrosswordLayout, List[str]]:
    """Build a sparse crossword arrangement with progress tracking."""
    cleaned = _sanitize_word_list(words)
    if not cleaned:
        return CrosswordLayout(clearance=clearance), []
    
    if max_width is not None or max_height is not None:
        max_dim = max(d for d in (max_width, max_height) if d is not None)
        filtered = []
        rejected = []
        for w in cleaned:
            if len(w) <= max_dim:
                filtered.append(w)
            else:
                rejected.append(w)
        cleaned = filtered

    # Longest-first spine still helps stability
    cleaned.sort(key=len, reverse=True)

    meta = _build_word_meta(cleaned)
    valid_words_set = set(cleaned)

    base_seed = seed if seed is not None else random.randrange(1 << 30)

    best_layout: Optional[CrosswordLayout] = None
    best_unplaced: List[str] = []
    # (placed_count, -components, -density_diff, -area)
    best_score: Tuple[int, int, float, int] = (-1, -9999, -9999.0, -999999)

    # Inner progress bar for attempts on a single file
    # leave=False hides this bar once the attempts for this file are complete
    pbar = tqdm(range(max(1, attempts)), desc="  Attempts", leave=False)

    for attempt in pbar:
        rng = random.Random(base_seed + attempt * 99991)

        # shuffle for diversity
        working = cleaned[:]
        if attempt > 0:
            rng.shuffle(working)

        layout, unplaced = _beam_attempt(
            working,
            meta,
            rng=rng,
            clearance=clearance,
            target_density=target_density,
            touch_penalty=touch_penalty,
            max_width=max_width,
            max_height=max_height,
            allow_islands=allow_islands,
            island_gap=island_gap,
            beam_width=beam_width,
            branch_factor=branch_factor,
            word_choices=word_choices,
            max_word_scan=max_word_scan,
            valid_words=valid_words_set,
        )

        placed_count = len(layout.placements)
        comps = layout.connected_components()
        density_diff = abs(layout.density() - target_density)
        score = (placed_count, -comps, -density_diff, -layout.area())

        if score > best_score:
            best_score = score
            best_layout = layout
            best_unplaced = unplaced
            # Show the current best placement count in the progress bar suffix
            pbar.set_postfix({"placed": f"{placed_count}/{len(cleaned)}"})

    assert best_layout is not None
    return best_layout, best_unplaced


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate sparse crosswords (improved generator).")
    parser.add_argument("--input_dir", help="Directory containing .tsv files (Word <tab> Hint)")
    parser.add_argument("--output_dir", help="Directory where .ipuz files will be saved")

    parser.add_argument("--attempts", type=int, default=50, help="Number of generation attempts / restarts")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--clearance", type=int, default=0, help="Distance between non-intersecting words")
    parser.add_argument("--touch-penalty", type=float, default=0.0, help="Penalty per perpendicular adjacency when --clearance=0 (0 = no penalty).")

    parser.add_argument("--density", type=float, default=0.8, help="Target density (0.0 - 1.0). Higher = more compact.")
    parser.add_argument("--width", type=int, default=23, help="Max width constraint")
    parser.add_argument("--height", type=int, default=23, help="Max height constraint")

    parser.add_argument("--islands", action="store_true", help="Enable disconnected word islands (defaults to False)")
    parser.add_argument("--island-gap", type=int, default=1, help="Gap size for islands")

    parser.add_argument("--beam-width", type=int, default=20, help="Beam width (1 = greedy; higher places more words)")
    parser.add_argument("--branch-factor", type=int, default=20, help="Top placements kept per expanded word")
    parser.add_argument("--word-choices", type=int, default=6, help="How many MRV words to branch on per state")
    parser.add_argument("--max-word-scan", type=int, default=100, help="Max words scanned per state expansion")

    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        print("Please provide --input_dir and --output_dir")
        return

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tsv_files = list(input_path.glob("*.tsv"))
    if not tsv_files:
        print(f"No .tsv files found in {args.input_dir}")
        return

    # Outer progress bar for the total list of files
    for tsv_file in tqdm(tsv_files, desc="Total Files"):
        output_filename = output_path / (tsv_file.stem + ".ipuz")

        word_hint_map: Dict[str, str] = {}
        word_structure_map: Dict[str, str] = {}

        try:
            with open(tsv_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 2:
                        continue
                    word_raw, hint = row[0], row[1].strip()
                    cleaned = _clean_word(word_raw)
                    if len(cleaned) >= 2:
                        word_hint_map[cleaned] = hint
                        word_structure_map[cleaned] = word_raw
        except Exception as e:
            tqdm.write(f"Error reading {tsv_file}: {e}")
            continue

        words_to_place = list(word_hint_map.keys())
        if not words_to_place:
            print("No valid words found.")
            continue

        layout, missing = generate_sparse_crossword(
            words_to_place,
            attempts=args.attempts,
            seed=args.seed,
            clearance=args.clearance,
            touch_penalty=args.touch_penalty,
            target_density=args.density,
            max_width=args.width,
            max_height=args.height,
            allow_islands=args.islands,
            island_gap=args.island_gap,
            beam_width=max(1, args.beam_width),
            branch_factor=max(1, args.branch_factor),
            word_choices=max(1, args.word_choices),
            max_word_scan=max(1, args.max_word_scan),
        )

        layout.save_ipuz(
            str(output_filename),
            title=tsv_file.stem.replace("_", " ").title(),
            clues_map=word_hint_map,
            structure_map=word_structure_map,
        )

        final_d = layout.density()

        # Use tqdm.write so log messages don't break the progress bar UI
        tqdm.write(
            f"Completed {tsv_file.name}: {len(layout.placements)}/{len(words_to_place)} placed.\n"
            f"Grid Size: {layout.width()}x{layout.height()} (Density: {final_d:.2f})"
        )
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            tqdm.write(f"Missing ({len(missing)}): {preview}{suffix}")

if __name__ == "__main__":
    main()
