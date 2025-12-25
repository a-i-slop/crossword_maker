from __future__ import annotations

import argparse
import csv
import collections
import itertools
import json
from pathlib import Path
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Any

Coord = Tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class Placement:
    """One placed word in the grid."""
    word: str
    x: int
    y: int
    direction: str  # 'H' or 'V'


@dataclass
class CrosswordLayout:
    """
    A sparse crossword-like word arrangement.
    Grid is stored as a dict from (x, y) -> letter. Any coordinate not in `grid` is a black cell.
    """
    clearance: int = 2
    grid: Dict[Coord, str] = field(default_factory=dict)

    # For each occupied cell, which directions already occupy the cell (H and/or V).
    cell_dirs: Dict[Coord, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))

    # For each occupied cell, which placement indices occupy the cell.
    cell_words: Dict[Coord, List[int]] = field(default_factory=lambda: collections.defaultdict(list))

    # Index for fast candidate generation: letter -> coordinates where that letter exists.
    letter_coords: Dict[str, List[Coord]] = field(default_factory=lambda: collections.defaultdict(list))

    placements: List[Placement] = field(default_factory=list)

    # Bounding box of current letters (min/max inclusive). If empty grid: maxx == -1.
    minx: int = 0
    maxx: int = -1
    miny: int = 0
    maxy: int = -1

    def bbox(self) -> Tuple[int, int, int, int]:
        """(minx, maxx, miny, maxy) for the occupied letters."""
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
        """Fraction of occupied letter cells within the tight bounding box."""
        return 0.0 if not self.grid else len(self.grid) / max(1, self.area())

    def iter_positions(self, word: str, x: int, y: int, direction: str) -> List[Coord]:
        if direction == "H":
            return [(x + i, y) for i in range(len(word))]
        if direction == "V":
            return [(x, y + i) for i in range(len(word))]
        raise ValueError("direction must be 'H' or 'V'")

    def _update_bbox(self, positions: Sequence[Coord]) -> None:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        if self.maxx == -1:
            self.minx, self.maxx = min(xs), max(xs)
            self.miny, self.maxy = min(ys), max(ys)
            return
        self.minx = min(self.minx, min(xs))
        self.maxx = max(self.maxx, max(xs))
        self.miny = min(self.miny, min(ys))
        self.maxy = max(self.maxy, max(ys))

    def can_place(
        self,
        word: str,
        x: int,
        y: int,
        direction: str,
        *,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> Optional[int]:
        """
        Check if a word can be placed.
        Returns the number of intersections if valid, else None.
        """
        positions = self.iter_positions(word, x, y, direction)

        # End caps must be black.
        if direction == "H":
            if (x - 1, y) in self.grid:
                return None
            if (x + len(word), y) in self.grid:
                return None
        else:  # 'V'
            if (x, y - 1) in self.grid:
                return None
            if (x, y + len(word)) in self.grid:
                return None

        intersections = 0
        for idx, pos in enumerate(positions):
            ch = word[idx]
            existing = self.grid.get(pos)

            if existing is not None:
                # Must match and must not overlap an existing word in the same direction.
                if existing != ch:
                    return None
                if direction in self.cell_dirs[pos]:
                    return None
                intersections += 1
                continue

            # New letter cell: enforce empty band in perpendicular direction.
            if direction == "H":
                for d in range(1, self.clearance + 1):
                    if (pos[0], pos[1] - d) in self.grid:
                        return None
                    if (pos[0], pos[1] + d) in self.grid:
                        return None
            else:  # 'V'
                for d in range(1, self.clearance + 1):
                    if (pos[0] - d, pos[1]) in self.grid:
                        return None
                    if (pos[0] + d, pos[1]) in self.grid:
                        return None

        # Optional bounding box constraint.
        if max_width is not None or max_height is not None:
            if not self.grid:
                new_minx = min(px for px, _ in positions)
                new_maxx = max(px for px, _ in positions)
                new_miny = min(py for _, py in positions)
                new_maxy = max(py for _, py in positions)
            else:
                new_minx = min(self.minx, min(px for px, _ in positions))
                new_maxx = max(self.maxx, max(px for px, _ in positions))
                new_miny = min(self.miny, min(py for _, py in positions))
                new_maxy = max(self.maxy, max(py for _, py in positions))

            new_w = new_maxx - new_minx + 1
            new_h = new_maxy - new_miny + 1
            if max_width is not None and new_w > max_width:
                return None
            if max_height is not None and new_h > max_height:
                return None

        return intersections

    def add_word(self, word: str, x: int, y: int, direction: str) -> None:
        """Mutates the layout by placing the word."""
        positions = self.iter_positions(word, x, y, direction)
        placement_index = len(self.placements)
        self.placements.append(Placement(word=word, x=x, y=y, direction=direction))

        for i, pos in enumerate(positions):
            ch = word[i]
            is_new_cell = pos not in self.grid
            self.grid[pos] = ch
            self.cell_dirs[pos].add(direction)
            self.cell_words[pos].append(placement_index)
            if is_new_cell:
                self.letter_coords[ch].append(pos)

        self._update_bbox(positions)

    def connected_components(self) -> int:
        """Number of connected components of placed words (connected via intersections)."""
        n = len(self.placements)
        if n == 0:
            return 0

        parent = list(range(n))
        rank = [0] * n

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for _, idxs in self.cell_words.items():
            if len(idxs) >= 2:
                base = idxs[0]
                for other in idxs[1:]:
                    union(base, other)

        return len({find(i) for i in range(n)})

    def render(
        self,
        *,
        pad: int = 1,
        pad_to_density: Optional[float] = None,
        empty: str = "#",
    ) -> List[str]:
        """
        Render to ASCII grid lines. '#' (default) means black.
        """
        if not self.grid:
            return []

        minx, maxx, miny, maxy = self.bbox()
        w = maxx - minx + 1
        h = maxy - miny + 1

        p = max(0, pad)
        if pad_to_density is not None and pad_to_density > 0:
            filled = len(self.grid)
            needed_area = filled / pad_to_density
            while (w + 2 * p) * (h + 2 * p) < needed_area:
                p += 1

        minx -= p
        maxx += p
        miny -= p
        maxy += p

        lines: List[str] = []
        for yy in range(miny, maxy + 1):
            row = []
            for xx in range(minx, maxx + 1):
                row.append(self.grid.get((xx, yy), empty))
            lines.append("".join(row))
        return lines

    def to_text(self, **render_kwargs) -> str:
        return "\n".join(self.render(**render_kwargs))

    def save_ipuz(
        self, 
        filename: str, 
        title: str = "Sparse Crossword", 
        clues_map: Optional[Dict[str, str]] = None,
        structure_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Save the layout to an .ipuz file (JSON format).
        Generates standard crossword numbering and extracted clues.
        
        :param clues_map: Dictionary mapping 'WORD' -> 'Hint text'.
        """
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

        # Numbering counter
        number = 1

        for r in range(h):
            for c in range(w):
                # Calculate original coordinates
                x = minx + c
                y = miny + r
                
                # Check if cell is black or letter
                if (x, y) not in self.grid:
                    puzzle_grid[r][c] = "#"
                    solution_grid[r][c] = "#"
                    continue
                
                # It is a letter
                letter = self.grid[(x, y)]
                solution_grid[r][c] = letter
                
                # Determine if this cell starts a word
                left_is_block = (x - 1, y) not in self.grid
                right_is_letter = (x + 1, y) in self.grid
                top_is_block = (x, y - 1) not in self.grid
                bottom_is_letter = (x, y + 1) in self.grid
                
                is_across = left_is_block and right_is_letter
                is_down = top_is_block and bottom_is_letter
                
                cell_val = 0  
                
                if is_across or is_down:
                    cell_val = number
                    number += 1
                    
                    if is_across:
                        # Reconstruct word
                        word_chars = []
                        cx = x
                        while (cx, y) in self.grid:
                            word_chars.append(self.grid[(cx, y)])
                            cx += 1
                        word_str = "".join(word_chars)
                        # Look up hint
                        hint_text = clues_map.get(word_str, f"Clue for {word_str}")
                        # Check if original word had spaces
                        orig = structure_map.get(word_str, "")
                        if " " in orig:
                            lengths = [len(part) for part in orig.split()]
                            hint_text = f"{hint_text} ({', '.join(map(str, lengths))})"
                        clues["Across"].append([cell_val, hint_text])

                    if is_down:
                        # Reconstruct word
                        word_chars = []
                        cy = y
                        while (x, cy) in self.grid:
                            word_chars.append(self.grid[(x, cy)])
                            cy += 1
                        word_str = "".join(word_chars)
                        # Look up hint
                        hint_text = clues_map.get(word_str, f"Clue for {word_str}")
                        # Check if original word had spaces
                        orig = structure_map.get(word_str, "")
                        if " " in orig:
                            lengths = [len(part) for part in orig.split()]
                            hint_text = f"{hint_text} ({', '.join(map(str, lengths))})"
                        clues["Down"].append([cell_val, hint_text])
                
                puzzle_grid[r][c] = cell_val

        ipuz_data = {
            "version": "http://ipuz.org/v2",
            "kind": ["http://ipuz.org/crossword#1"],
            "dimensions": {"width": w, "height": h},
            "puzzle": puzzle_grid,
            "solution": solution_grid,
            "clues": clues,
            "title": title,
            "origin": "Generated by Python Sparse Crossword Generator"
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(ipuz_data, f, indent=2)
        print(f"Saved {filename}")


def _clean_word(w: str) -> str:
    """Standardize words: uppercase, only A-Z."""
    return re.sub(r"[^A-Za-z]", "", w.strip()).upper()


def _sanitize_word_list(words: Iterable[str]) -> List[str]:
    """
    Keep A–Z only, uppercase, drop duplicates, drop very short words (<2).
    """
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


def generate_sparse_crossword(
    words: Sequence[str],
    *,
    attempts: int = 300,
    seed: Optional[int] = None,
    clearance: int = 2,
    target_density: float = 0.35,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    allow_islands: bool = True,
    island_gap: int = 3,
) -> Tuple[CrosswordLayout, List[str]]:
    """
    Build a sparse crossword arrangement from a list of words.
    Returns: (best_layout, unplaced_words)
    """
    # Sanitize input list
    cleaned = _sanitize_word_list(words)
    
    if not cleaned:
        return CrosswordLayout(clearance=clearance), []

    base_seed = seed if seed is not None else random.randrange(1 << 30)
    best_layout: Optional[CrosswordLayout] = None
    best_score: Optional[Tuple[int, int, float, int]] = None
    best_unplaced: List[str] = []

    side_cycle = itertools.cycle(["RIGHT", "DOWN", "LEFT", "UP"])

    for attempt in range(max(1, attempts)):
        rng = random.Random(base_seed + attempt * 99991)

        buckets: Dict[int, List[str]] = collections.defaultdict(list)
        for w in cleaned:
            buckets[len(w)].append(w)
        ordered: List[str] = []
        for L in sorted(buckets.keys(), reverse=True):
            bucket = buckets[L][:]
            rng.shuffle(bucket)
            ordered.extend(bucket)

        layout = CrosswordLayout(clearance=clearance)

        layout.add_word(ordered[0], 0, 0, "H")
        remaining: List[str] = ordered[1:]

        stalls = 0
        while remaining and stalls < len(remaining) + 5:
            scored: List[Tuple[int, int, float, str]] = []
            for w in remaining:
                pot = 0
                for ch in set(w):
                    pot += len(layout.letter_coords.get(ch, []))
                scored.append((pot, len(w), rng.random(), w))
            scored.sort(reverse=True)
            word = scored[0][3]

            candidates: List[Tuple[int, int, int, str]] = []  # (intersections, x, y, dir)
            for i, ch in enumerate(word):
                for gx, gy in layout.letter_coords.get(ch, []):
                    hx, hy = gx - i, gy
                    inter = layout.can_place(word, hx, hy, "H", max_width=max_width, max_height=max_height)
                    if inter is not None and inter > 0:
                        candidates.append((inter, hx, hy, "H"))

                    vx, vy = gx, gy - i
                    inter = layout.can_place(word, vx, vy, "V", max_width=max_width, max_height=max_height)
                    if inter is not None and inter > 0:
                        candidates.append((inter, vx, vy, "V"))

            if not candidates:
                stalls += 1
                remaining.remove(word)
                remaining.append(word)

                if allow_islands and stalls >= len(remaining) and remaining:
                    word = max(remaining, key=len)
                    side = next(side_cycle)

                    minx, maxx, miny, maxy = layout.bbox()
                    gap = island_gap + layout.clearance + 1

                    if side == "RIGHT":
                        x, y, d = maxx + gap, miny, "H"
                    elif side == "LEFT":
                        x, y, d = minx - gap - len(word) + 1, miny, "H"
                    elif side == "DOWN":
                        x, y, d = minx, maxy + gap, "V"
                    else:  # UP
                        x, y, d = minx, miny - gap - len(word) + 1, "V"

                    inter = layout.can_place(word, x, y, d, max_width=max_width, max_height=max_height)

                    if inter is None:
                        placed = False
                        for _ in range(250):
                            d = rng.choice(["H", "V"])
                            minx, maxx, miny, maxy = layout.bbox()
                            gap = island_gap + layout.clearance + 1
                            if d == "H":
                                x = rng.choice([maxx + gap, minx - gap - len(word) + 1])
                                y = rng.randint(miny - gap, maxy + gap)
                            else:
                                x = rng.randint(minx - gap, maxx + gap)
                                y = rng.choice([maxy + gap, miny - gap - len(word) + 1])

                            if layout.can_place(word, x, y, d, max_width=max_width, max_height=max_height) is not None:
                                layout.add_word(word, x, y, d)
                                remaining.remove(word)
                                stalls = 0
                                placed = True
                                break
                        if not placed:
                            break
                    else:
                        layout.add_word(word, x, y, d)
                        remaining.remove(word)
                        stalls = 0

                continue

            best_cand: Optional[Tuple[int, int, str]] = None
            best_cand_score: Optional[float] = None

            for inter, x, y, d in candidates:
                positions = layout.iter_positions(word, x, y, d)

                new_minx = min(layout.minx, min(px for px, _ in positions))
                new_maxx = max(layout.maxx, max(px for px, _ in positions))
                new_miny = min(layout.miny, min(py for _, py in positions))
                new_maxy = max(layout.maxy, max(py for _, py in positions))
                new_area = (new_maxx - new_minx + 1) * (new_maxy - new_miny + 1)

                new_letters = sum(1 for p in positions if p not in layout.grid)
                new_filled = len(layout.grid) + new_letters
                density = new_filled / max(1, new_area)

                sparsity_bonus = (target_density - density)
                area_growth = new_area - layout.area()
                new_w = new_maxx - new_minx + 1
                new_h = new_maxy - new_miny + 1
                aspect_ratio_penalty = (new_w - new_h)**2
                score = (inter * 100
                        + sparsity_bonus * 800
                        - area_growth * 0.05
                        - aspect_ratio_penalty * 10.0
                        + rng.random() * 0.01)

                if best_cand_score is None or score > best_cand_score:
                    best_cand_score = score
                    best_cand = (x, y, d)

            if best_cand is not None:
                layout.add_word(word, best_cand[0], best_cand[1], best_cand[2])
                remaining.remove(word)
                stalls = 0
            else:
                stalls += 1

        placed_words = {p.word for p in layout.placements}
        unplaced = [w for w in cleaned if w not in placed_words]

        comps = layout.connected_components()
        dens = layout.density() if layout.area() else 1.0
        over = max(0.0, dens - target_density)
        score_tuple = (len(layout.placements), -comps, -over, -layout.area())

        if best_score is None or score_tuple > best_score:
            best_score = score_tuple
            best_layout = layout
            best_unplaced = unplaced

    assert best_layout is not None
    return best_layout, best_unplaced


def main():
    parser = argparse.ArgumentParser(description="Batch generate sparse crosswords from a folder of TSV files.")
    
    # Input/Output Folders
    parser.add_argument("--input_dir", help="Directory containing .tsv files")
    parser.add_argument("--output_dir", help="Directory where .ipuz files will be saved")
    
    # Generator Parameters
    parser.add_argument("--attempts", type=int, default=3000, help="Number of generation attempts per file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--clearance", type=int, default=1, help="Distance between non-intersecting words")
    parser.add_argument("--density", type=float, default=0.35, help="Target fill density (0.0 - 1.0)")
    parser.add_argument("--width", type=int, default=25, help="Max width constraint")
    parser.add_argument("--height", type=int, default=25, help="Max height constraint")
    
    # Islands
    parser.add_argument("--no-islands", action="store_true", help="Disable disconnected word islands")
    parser.add_argument("--island-gap", type=int, default=3, help="Gap size for islands")

    args = parser.parse_args()

    # Ensure output directory exists
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .tsv files
    tsv_files = list(input_path.glob("*.tsv"))
    
    if not tsv_files:
        print(f"No .tsv files found in {args.input_dir}")
        return

    print(f"Found {len(tsv_files)} files to process.")

    for tsv_file in tsv_files:
        print(f"\nProcessing: {tsv_file.name}")
        
        # Prepare output filename: strip .tsv and add .ipuz
        output_filename = output_path / (tsv_file.stem + ".ipuz")
        
        # Read and parse TSV
        word_hint_map = {}
        word_structure_map = {}
        with open(tsv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue
                word_raw, hint = row[0], row[1].strip()
                cleaned_word = _clean_word(word_raw)
                if len(cleaned_word) >= 2:
                    word_hint_map[cleaned_word] = hint
                    word_structure_map[cleaned_word] = word_raw

        words_to_place = list(word_hint_map.keys())
        if not words_to_place:
            print(f"Skipping {tsv_file.name}: No valid words found.")
            continue

        # Generate the layout
        layout, missing = generate_sparse_crossword(
            words_to_place,
            attempts=args.attempts,
            seed=args.seed,
            clearance=args.clearance,
            target_density=args.density,
            max_width=args.width,
            max_height=args.height,
            allow_islands=not args.no_islands,
            island_gap=args.island_gap,
        )

        # Save result using the stem name as the title
        layout.save_ipuz(
            str(output_filename), 
            title=tsv_file.stem.replace('_', ' ').title(), 
            clues_map=word_hint_map,
            structure_map=word_structure_map,
        )
        print(f"Placed {len(layout.placements)} words. Missing: {len(missing)}")

if __name__ == "__main__":
    main()
