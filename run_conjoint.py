from __future__ import annotations

from pathlib import Path

from src.conjoint.fit_conjoint import run_conjoint


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "raw" / "flipkart_laptops.csv"
    output_dir = project_root / "data" / "outputs" / "conjoint"

    run_conjoint(csv_path, output_dir)

    print("Conjoint pipeline completed.")
    print(f"Input CSV: {csv_path}")
    print(f"Outputs directory: {output_dir}")


if __name__ == "__main__":
    main()
