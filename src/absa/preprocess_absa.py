from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AbsaPaths:
    project_root: Path
    part1_input: Path
    part2_input: Path
    combined_output: Path


def default_paths(project_root: Path) -> AbsaPaths:
    interim_dir = project_root / "data" / "interim"
    return AbsaPaths(
        project_root=project_root,
        part1_input=interim_dir / "absa_output_electric_tb_p1.csv",
        part2_input=interim_dir / "absa_output_electric_tb_p2.csv",
        combined_output=interim_dir / "absa_output_electric_tb_combined.csv",
    )


def combine_absa_outputs(paths: AbsaPaths) -> pd.DataFrame:
    frames = [pd.read_csv(paths.part1_input), pd.read_csv(paths.part2_input)]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(paths.combined_output, index=False)
    return combined
