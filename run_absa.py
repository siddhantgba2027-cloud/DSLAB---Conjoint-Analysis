from __future__ import annotations

from pathlib import Path

from src.absa.preprocess_absa import combine_absa_outputs, default_paths as preprocess_default_paths
from src.absa.score_absa import default_paths as score_default_paths
from src.absa.score_absa import score_absa


def main() -> None:
    project_root = Path(__file__).resolve().parent
    preprocess_paths = preprocess_default_paths(project_root)
    score_paths = score_default_paths(project_root)

    combine_absa_outputs(preprocess_paths)
    outputs = score_absa(score_paths)

    print("ABSA pipeline completed.")
    print(f"Combined input: {preprocess_paths.combined_output}")
    print(f"Final scored output: {score_paths.scored_output}")
    print(f"Aspect summary: {score_paths.aspect_summary_output}")
    print(f"Review summary: {score_paths.review_summary_output}")
    print(f"Rows processed: {len(outputs['scored'])}")


if __name__ == "__main__":
    main()
