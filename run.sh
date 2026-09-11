#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Use the current Python by default; allow a custom interpreter when needed.
PYTHON_BIN="${PYTHON_BIN:-python}"

run_notebook() {
    echo "Running $1 ..."
    "$PYTHON_BIN" -m jupyter nbconvert \
        --to notebook \
        --execute "$1" \
        --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.allow_errors=False
}

run_notebook code0.ipynb
run_notebook code1.ipynb
run_notebook code2.ipynb
run_notebook code3.ipynb
run_notebook code4.ipynb
run_notebook code5.ipynb
run_notebook code6.ipynb

# Generate the intermediate results required by the figure notebooks.
run_notebook book0.ipynb

run_notebook book1.ipynb
run_notebook book2.ipynb
run_notebook book3.ipynb
run_notebook book4.ipynb
run_notebook book5.ipynb
run_notebook book6.ipynb
run_notebook book7.ipynb
run_notebook book8.ipynb
run_notebook book9.ipynb

echo "All notebooks completed successfully."