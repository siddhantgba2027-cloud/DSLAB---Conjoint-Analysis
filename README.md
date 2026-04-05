# Project Submission

This repository is organized so the instructor can run one command per analysis workflow.

## Project Structure

- `run_absa.py`: runs the ABSA scoring pipeline from the provided extracted ABSA CSV files.
- `run_conjoint.py`: runs the laptop conjoint analysis and exports tables and plots.
- `data/raw/`: original raw datasets.
- `data/interim/`: intermediate ABSA extraction outputs used as pipeline inputs.
- `data/external/`: reference checkpoint files retained from the original work.
- `data/outputs/`: generated outputs for submission.
- `notebooks/archive/`: original notebooks kept as references only.

## Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install -r requirements.txt` was attempted earlier with the old dependency pins, run:

```bash
pip uninstall -y tokenizers transformers sentence-transformers gensim
pip install -r requirements.txt
```

## Run

Run the ABSA workflow:

```bash
python run_absa.py
```

Run the conjoint workflow:

```bash
python run_conjoint.py
```

## Output Locations

- ABSA outputs are written to `data/outputs/absa/`
- Conjoint outputs are written to `data/outputs/conjoint/`

## Notes

- The scraping notebook was intentionally excluded from the submission flow.
- The ABSA submission pipeline starts from the existing extracted ABSA CSV files in `data/interim/` to keep execution reliable for grading.
- The original notebooks are preserved in `notebooks/archive/` for transparency.
- The refactored submission does not require `pyabsa` or spaCy to run, because the ABSA extraction outputs are already included in `data/interim/`.
