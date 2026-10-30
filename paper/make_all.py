"""
Regenerate the result figures and the table bodies of the classical paper.

    python paper/make_all.py

Inputs are the result files already in the repository (results/paper/ and
studies/deepvariant_low_coverage/results/). No dataset, checkpoint or GPU is needed.
"""

import runpy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"


def main():
    runpy.run_path(str(SCRIPTS / "make_figures.py"), run_name="__main__")  # Figures 3 and 4
    runpy.run_path(str(SCRIPTS / "make_tables.py"), run_name="__main__")   # Tables 2-4 and 6-9


if __name__ == "__main__":
    main()
