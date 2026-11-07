#!/usr/bin/env bash
# Build the cha2022 tool environment without Docker, for the "user" sandbox (HPC nodes, cloud containers).
#
#   sudo env/setup_local.sh [prefix]      # default prefix /opt/mathbac
#
# Installs a Python 3.8 interpreter and the capsule's pinned packages under <prefix> (readable by the
# agent user, unlike ~/.local), the released code at /opt/unified_fast, and the tool wrappers at
# <prefix>/tools. GHECOM is not included: its source needs registration at https://pdbj.org/ghecom.
# TensorFlow 2.3 wheels exist for x86_64 Linux only.
set -euo pipefail
PREFIX="${1:-/opt/mathbac}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIFIED_FAST_COMMIT=04b567b4ef85dc916209510bd830b1d081e5beb5

command -v uv >/dev/null || pip install --quiet uv
mkdir -p "$PREFIX"
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
uv python install 3.8
rm -rf "$PREFIX/env"
uv venv -p 3.8 "$PREFIX/env"
printf 'numpy==1.21.2\nscipy==1.4.1\n' > "$PREFIX/overrides.txt"   # TF 2.3 declares numpy<1.19; the capsule ran 1.21.2
uv pip install -p "$PREFIX/env/bin/python" --override "$PREFIX/overrides.txt" -r "$REPO/env/requirements-capsule.txt"

if [ ! -d /opt/unified_fast/.git ]; then
  git clone https://github.com/jessqma/unified_fast /opt/unified_fast
fi
git -C /opt/unified_fast checkout -q "$UNIFIED_FAST_COMMIT"

mkdir -p "$PREFIX/tools"
cp "$REPO"/env/tools/*.sh "$REPO"/env/tools/*.py "$PREFIX/tools/"
chmod 755 "$PREFIX/tools"/*.sh
chmod 644 "$PREFIX/tools"/*.py
chmod -R a+rX "$PREFIX" /opt/unified_fast

"$PREFIX/env/bin/python" -c "import numpy, scipy, sklearn, tensorflow, prody, networkx, GraphRicciCurvature, xgboost; print('environment ok')"
