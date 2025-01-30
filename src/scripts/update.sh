set -e
nbdev_update
nbdev_export
nbdev_clean
nbdev_bump_version
python -m pip install -e .
find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -exec rm -rf {} +
