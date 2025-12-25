from __future__ import annotations
import argparse
import csv
import collections
import itertools
import json
import math
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
    
    # Bounding box
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
    
    def centroid(self) -> Tuple[float, float]:
        """Returns the center of mass of the current grid."""
        if not self.grid:
            return (0.0, 0.0)
        # Fast approximation using bbox center is usually sufficient and faster than averaging all points
        return ((self.minx + self.maxx) / 2.0, (self.miny + self.maxy) / 2.0)

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
        
        # 1. Bounds check (End caps must be empty)
        if direction == "H":
            if (x - 1, y) in self.grid: return None
            if (x + len(word), y) in self.grid: return None
        else:
            if (x, y - 1) in self.grid: return None
            if (x, y + len(word)) in self.grid: return None

        intersections = 0
        
        # 2. Collision and Clearance check
        for idx, pos in enumerate(positions):
            ch = word[idx]
            existing = self.grid.get(pos)
            
            if existing is not None:
                # Intersection logic
                if existing != ch:
                    return None
                if direction in self.cell_dirs[pos]:
                    return None # Already occupied in this direction
                intersections += 1
            else:
                # New cell logic: Check Clearance (perpendicular neighbors)
                if direction == "H":
                    for d in range(1, self.clearance + 1):
                        if (pos[0], pos[1] - d) in self.grid: return None
                        if (pos[0], pos[1] + d) in self.grid: return None
                else: # V
                    for d in range(1, self.clearance + 1):
                        if (pos[0] - d, pos[1]) in self.grid: return None
                        if (pos[0] + d, pos[1]) in self.grid: return None

        # 3. Global Size Constraint check
        if max_width is not None or max_height is not None:
            if not self.grid:
                new_w = positions[-1][0] - positions[0][0] + 1
                new_h = positions[-1][1] - positions[0][1] + 1
            else:
                current_xs = [p[0] for p in positions]
                current_ys = [p[1] for p in positions]
                cand_minx = min(self.minx, min(current_xs))
                cand_maxx = max(self.maxx, max(current_xs))
                cand_miny = min(self.miny, min(current_ys))
                cand_maxy = max(self.maxy, max(current_ys))
                new_w = cand_maxx - cand_minx + 1
                new_h = cand_maxy - cand_miny + 1
            
            if max_width is not None and new_w > max_width: return None
            if max_height is not None and new_h > max_height: return None

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
        
        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        for _, idxs in self.cell_words.items():
            if len(idxs) >= 2:
                base = idxs[0]
                for other in idxs[1:]:
                    union(base, other)
        
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
                
                # Check for start of word
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
                        # Reconstruct word
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
                        # Reconstruct word
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
            "origin": "Generated by Python Sparse Crossword Generator"
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
        if len(w) < 2: continue
        if w in seen: continue
        seen.add(w)
        out.append(w)
    return out

def generate_sparse_crossword(
    words: Sequence[str],
    *,
    attempts: int = 100,
    seed: Optional[int] = None,
    clearance: int = 2,
    target_density: float = 0.5,  # Re-introduced parameter
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    allow_islands: bool = True,
    island_gap: int = 3,
) -> Tuple[CrosswordLayout, List[str]]:
    """
    Build a sparse crossword arrangement.
    Optimized for structure (spine-first) + configurable density target.
    """
    cleaned = _sanitize_word_list(words)
    if not cleaned:
        return CrosswordLayout(clearance=clearance), []

    # Sort strictly by length to form a strong spine
    cleaned.sort(key=len, reverse=True)

    base_seed = seed if seed is not None else random.randrange(1 << 30)
    
    best_layout: Optional[CrosswordLayout] = None
    # Score tuple: (placed_count, -components, -density_diff, -area)
    # We want to minimize the difference from target density
    best_result_score: Tuple[int, int, float, int] = (-1, -999, -999.0, -999999)
    best_unplaced: List[str] = []

    for attempt in range(max(1, attempts)):
        rng = random.Random(base_seed + attempt * 99991)
        
        # Keep longest 3 words at start (spine), shuffle the rest slightly for variation
        working_list = cleaned[:]
        if attempt > 0 and len(working_list) > 3:
            spine = working_list[:3]
            tail = working_list[3:]
            rng.shuffle(tail)
            working_list = spine + tail

        layout = CrosswordLayout(clearance=clearance)
        
        # Place first word
        layout.add_word(working_list[0], 0, 0, "H")
        remaining = working_list[1:]
        
        stalls = 0
        max_stalls = len(remaining) * 2 

        while remaining and stalls < max_stalls:
            # Heuristic: prioritize words that share letters with existing grid
            scored_words = []
            for w in remaining:
                overlap_potential = 0
                for ch in set(w):
                    overlap_potential += len(layout.letter_coords.get(ch, []))
                scored_words.append((overlap_potential + rng.random(), w))
            
            scored_words.sort(reverse=True)
            
            # Search top candidates
            search_depth = min(len(remaining), 5)
            candidates: List[Tuple[float, int, int, str, str]] = [] # (score, x, y, dir, word)

            for _, word in scored_words[:search_depth]:
                # Find all valid positions
                for i, ch in enumerate(word):
                    for gx, gy in layout.letter_coords.get(ch, []):
                        # Check H and V
                        for direction, (cx, cy) in [("H", (gx - i, gy)), ("V", (gx, gy - i))]:
                            inter = layout.can_place(word, cx, cy, direction, max_width=max_width, max_height=max_height)
                            if inter is not None and inter > 0:
                                
                                # --- SCORING CALCULATION ---
                                positions = layout.iter_positions(word, cx, cy, direction)
                                pxs = [p[0] for p in positions]
                                pys = [p[1] for p in positions]
                                
                                # New Dimensions
                                new_minx = min(layout.minx, min(pxs))
                                new_maxx = max(layout.maxx, max(pxs))
                                new_miny = min(layout.miny, min(pys))
                                new_maxy = max(layout.maxy, max(pys))
                                new_w = new_maxx - new_minx + 1
                                new_h = new_maxy - new_miny + 1
                                new_area = new_w * new_h
                                area_growth = new_area - layout.area()

                                # New Density
                                new_letters = sum(1 for p in positions if p not in layout.grid)
                                new_filled = len(layout.grid) + new_letters
                                new_density = new_filled / max(1, new_area)

                                # Centroid / Gravity
                                grid_cx, grid_cy = layout.centroid()
                                word_mid_x = (min(pxs) + max(pxs)) / 2.0
                                word_mid_y = (min(pys) + max(pys)) / 2.0
                                dist_from_center = math.sqrt((word_mid_x - grid_cx)**2 + (word_mid_y - grid_cy)**2)

                                # Metric 1: Connections (Squared is better)
                                score_inter = inter ** 2 * 10.0
                                
                                # Metric 2: Density Target
                                # We penalize deviation from the user's requested density.
                                # Weight needs to be high to counteract area growth penalties if user wants low density.
                                score_density = -abs(new_density - target_density) * 100.0

                                # Metric 3: Shape/Area
                                # If we are ALREADY denser than target, area growth is fine (penalty reduced).
                                # If we are sparser than target, area growth is bad (penalty increased).
                                area_penalty_factor = 0.25
                                if new_density < target_density:
                                    area_penalty_factor = 0.5  # Punish growth if we are too sparse
                                else:
                                    area_penalty_factor = 0.1  # Allow growth if we are too dense

                                score_area = -area_growth * area_penalty_factor
                                score_gravity = -dist_from_center * 1.5
                                score_aspect = -(new_w - new_h)**2 * 10.0

                                total_score = score_inter + score_density + score_area + score_gravity + score_aspect
                                candidates.append((total_score, cx, cy, direction, word))

            if candidates:
                # Pick best
                candidates.sort(key=lambda x: x[0], reverse=True)
                best = candidates[0]
                layout.add_word(best[4], best[1], best[2], best[3])
                remaining.remove(best[4])
                stalls = 0
            else:
                # Island fallback
                if allow_islands:
                    word = max(remaining, key=len)
                    placed_island = False
                    minx, maxx, miny, maxy = layout.bbox()
                    gap = island_gap + layout.clearance
                    
                    # Try placing near edges
                    attempts_island = [
                        (maxx + gap, miny, "V"), (minx - gap, miny, "V"),
                        (minx, maxy + gap, "H"), (minx, miny - gap, "H")
                    ]
                    for ix, iy, idir in attempts_island:
                        if layout.can_place(word, ix, iy, idir, max_width=max_width, max_height=max_height) is not None:
                            layout.add_word(word, ix, iy, idir)
                            remaining.remove(word)
                            placed_island = True
                            stalls = 0
                            break
                    
                    if not placed_island:
                        stalls += 1
                        rng.shuffle(remaining)
                else:
                    stalls += 1
                    rng.shuffle(remaining)

        # End of Attempt - Compare against best
        placed_count = len(layout.placements)
        comps = layout.connected_components()
        
        final_density = layout.density()
        density_diff = abs(final_density - target_density)
        
        # Priority: Placed Count > Fewest Islands > Closest to Target Density > Smallest Area
        score_tuple = (placed_count, -comps, -density_diff, -layout.area())
        
        if score_tuple > best_result_score:
            best_result_score = score_tuple
            best_layout = layout
            best_unplaced = remaining

    assert best_layout is not None
    return best_layout, best_unplaced

def main():
    parser = argparse.ArgumentParser(description="Batch generate sparse crosswords.")
    parser.add_argument("--input_dir", help="Directory containing .tsv files (Word <tab> Hint)")
    parser.add_argument("--output_dir", help="Directory where .ipuz files will be saved")
    parser.add_argument("--attempts", type=int, default=100, help="Number of generation attempts")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--clearance", type=int, default=1, help="Distance between non-intersecting words")
    parser.add_argument("--density", type=float, default=1.0, help="Target density (0.0 - 1.0). Higher = more compact.")
    parser.add_argument("--width", type=int, default=30, help="Max width constraint")
    parser.add_argument("--height", type=int, default=30, help="Max height constraint")
    parser.add_argument("--islands", action="store_true", help="Enable disconnected word islands (defaults to False)")
    parser.add_argument("--island-gap", type=int, default=1, help="Gap size for islands")
    
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

    print(f"Found {len(tsv_files)} files to process.")
    
    for tsv_file in tsv_files:
        print(f"\nProcessing: {tsv_file.name}")
        output_filename = output_path / (tsv_file.stem + ".ipuz")
        
        word_hint_map = {}
        word_structure_map = {}
        
        try:
            with open(tsv_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 2: continue
                    word_raw, hint = row[0], row[1].strip()
                    cleaned = _clean_word(word_raw)
                    if len(cleaned) >= 2:
                        word_hint_map[cleaned] = hint
                        word_structure_map[cleaned] = word_raw
        except Exception as e:
            print(f"Error reading {tsv_file}: {e}")
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
            target_density=args.density,
            max_width=args.width,
            max_height=args.height,
            allow_islands=args.islands,
            island_gap=args.island_gap,
        )

        # Save result using the stem name as the title
        layout.save_ipuz(
            str(output_filename), 
            title=tsv_file.stem.replace('_', ' ').title(), 
            clues_map=word_hint_map,
            structure_map=word_structure_map,
        )
        
        final_d = layout.density()
        print(f"Placed {len(layout.placements)}/{len(words_to_place)} words.")
        print(f"Grid Size: {layout.width()}x{layout.height()} (Density: {final_d:.2f})")
        if missing:
            print(f"Missing: {', '.join(missing[:5])}...")

if __name__ == "__main__":
    main()
