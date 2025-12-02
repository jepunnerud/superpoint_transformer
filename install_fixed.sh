#!/usr/bin/env bash
set -euo pipefail

########## Local variables ##########
PROJECT_NAME="spt_test"
PYTHON="3.8"
TORCH="2.2.0"
CUDA_SUPPORTED=("11.8" "12.1")
#####################################

log() { printf "\n\n⭐ %s\n\n" "$*"; }

# --- Move to script directory ---
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "_____________________________________________"
echo
echo "         🧩 Superpoint Transformer 🤖        "
echo "                  Installer                  "
echo
echo "_____________________________________________"
echo

# --- Prefer IDUN's global conda; fallback to user installs ---
detect_conda_sh() {
  local candidates=(
    "/cluster/apps/eb/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  )
  for c in "${candidates[@]}"; do
    [[ -f "$c" ]] && { echo "$c"; return 0; }
  done
  return 1
}

# --- Assume modules (CUDA + GCC) are already loaded by the user ---
# We only *check* and record compilers here, no `module load` calls.

# Resolve active gcc/g++
export CC="$(command -v gcc || true)"
export CXX="$(command -v g++ || true)"
if [[ -z "${CC}" || -z "${CXX}" ]]; then
  echo "Error: gcc/g++ not found on PATH. Load a GCC 11 module first (e.g. GCC/11.3.1)."
  exit 1
fi

GCC_MAJOR="$("$CC" -dumpversion | cut -d. -f1 || echo "")"
if [[ "$GCC_MAJOR" != "11" ]]; then
  echo "Warning: GCC 11.x recommended, but found: $("$CC" -dumpfullversion 2>/dev/null || "$CC" -dumpversion)"
  echo "Tip: run 'module avail GCC' and 'module load GCC/11.3.1' (or similar) before rerunning."
fi

echo "Using CC=$CC"
echo "Using CXX=$CXX"

# --- Detect CUDA version via nvcc ---
log "Detecting CUDA (nvcc)"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Error: nvcc not found. Load a CUDA module (e.g., CUDA/12.1.1) and retry."
  exit 1
fi

CUDA_VERSION="$(nvcc --version | grep -o 'release [0-9]\+\.[0-9]\+' | awk '{print $2}')"
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION##*.}"

echo "Found CUDA: ${CUDA_VERSION}"

# Check support
supported=false
for v in "${CUDA_SUPPORTED[@]}"; do
  [[ "$v" == "$CUDA_VERSION" ]] && supported=true && break
done
if [[ "$supported" != "true" ]]; then
  echo "Found CUDA ${CUDA_VERSION}, but supported are: ${CUDA_SUPPORTED[*]}"
  exit 1
fi

# Map CUDA version to wheel tag (cu121 / cu118)
case "${CUDA_VERSION}" in
  12.1*) WHEEL_CUDA_TAG="cu121" ;;
  11.8*) WHEEL_CUDA_TAG="cu118" ;;
  *) echo "Unknown CUDA wheel tag for ${CUDA_VERSION}"; exit 1 ;;
esac
echo "Using wheel CUDA tag: ${WHEEL_CUDA_TAG}"

# --- Source conda.sh ---
log "Sourcing conda"
CONDA_SH="$(detect_conda_sh)" || {
  echo "Could not find conda.sh. Install/Load conda and retry."
  exit 1
}
# shellcheck source=/dev/null
source "$CONDA_SH"

# --- Create & activate env ---
log "Creating conda environment '${PROJECT_NAME}' (python=${PYTHON})"
conda create --name "${PROJECT_NAME}" "python=${PYTHON}" -y
conda activate "${PROJECT_NAME}"

# Re-export compilers from the active shell (after conda activation)
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
echo "gcc: $($CC -v 2>&1 | tail -1)"
echo "g++: $($CXX -v 2>&1 | tail -1)"
echo "nvcc: $(nvcc --version | grep 'release' || true)"

# Avoid pip caching broken wheels during this run
export PIP_NO_CACHE_DIR=1

# --- Install Python & system build deps in a robust order ---
log "Installing conda and pip dependencies"

# Basic conda tools
conda install -y pip nb_conda_kernels

# Build tooling needed for native extensions (FRNN, pgeof, cut-pursuit, grid-graph)
conda install -y -c conda-forge cmake ninja pybind11 libstdcxx-ng

# Upgrade pip toolchain
pip install --upgrade pip setuptools wheel

# Modern build backend used by some projects
pip install scikit-build-core

# nanobind for point_geometric_features / pgeof
pip install "nanobind>=1.8.0"


# Jupyter stack
pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
pip install "notebook>=5.3" "ipywidgets>=7.5"
pip install ipykernel

# PyTorch matching CUDA
log "Installing PyTorch ${TORCH} for ${WHEEL_CUDA_TAG}"
pip install "torch==${TORCH}" torchvision --index-url "https://download.pytorch.org/whl/${WHEEL_CUDA_TAG}"

# PyG matching torch & CUDA
pip install torchmetrics==0.11.4
pip install pyg_lib torch_scatter torch_cluster -f "https://data.pyg.org/whl/torch-${TORCH}+${WHEEL_CUDA_TAG}.html"
pip install torch_geometric==2.3.0

# Misc packages
pip install matplotlib plotly==5.9.0 seaborn numba h5py plyfile colorhash pytorch-lightning \
           pyrootutils hydra-colorlog hydra-submitit-launcher "rich<=14.0" torch_tb_profiler \
           wandb open3d gdown ipyfilechooser

# --- Install native deps (clean builds) ---
log "Installing FRNN"
rm -rf src/dependencies/FRNN || true
git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN

# prefix_sum
pushd src/dependencies/FRNN/external/prefix_sum >/dev/null
python setup.py clean --all
python setup.py install
popd >/dev/null

# FRNN core
pushd src/dependencies/FRNN >/dev/null
python setup.py clean --all
python setup.py install
popd >/dev/null

# --- Point Geometric Features (requires build toolchain) ---
log "Installing Point Geometric Features"

# nanobind required for CMake config in point_geometric_features
pip install "nanobind>=1.8.0"

# Clean old caches to avoid broken cmake attempts
rm -rf build/cp38-cp38-linux_x86_64 || true

# Install fixed known-good commit without build isolation
pip install --no-build-isolation \
    git+https://github.com/drprojects/point_geometric_features.git@4102aa9

# --- Parallel Cut-Pursuit & Grid Graph ---
log "Installing Parallel Cut-Pursuit & Grid Graph"
rm -rf src/dependencies/parallel_cut_pursuit src/dependencies/grid_graph || true
git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git src/dependencies/parallel_cut_pursuit
git clone https://gitlab.com/1a7r0ch3/grid-graph.git src/dependencies/grid_graph

# Ensure previous builds are gone if the script is rerun
rm -rf src/dependencies/parallel_cut_pursuit/build src/dependencies/parallel_cut_pursuit/*.egg-info || true
rm -rf src/dependencies/grid_graph/build src/dependencies/grid_graph/*.egg-info || true

python scripts/setup_dependencies.py build_ext

# --- Install laspy laz backend ---
log "Installing laspy laz backend (laszip)"
pip install laszip==0.2.3

echo
echo "🚀 Successfully installed SPT into conda env '${PROJECT_NAME}'"
