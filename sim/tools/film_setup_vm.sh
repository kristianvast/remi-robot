#!/usr/bin/env bash
# Provisions elegoo-sim-2 for the film -> Gaussian splat job.
# Run on the VM: bash film_setup_vm.sh > ~/film/setup.log 2>&1
# Installs ffmpeg + COLMAP (apt), uv, and nv-tlabs/3dgrut with a venv-local CUDA 12.8 toolkit.
set -euxo pipefail
mkdir -p ~/film
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg colmap build-essential wget git \
  libgl1-mesa-dev libglib2.0-0 unzip
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~
[ -d 3dgrut ] || git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
git log -1 --format='3dgrut commit %H %cd'
FORCE_LOCAL_CUDA=1 CUDA_VERSION=12 ./scripts/create_venv_cuda.sh
source .venv/bin/activate
./install_env_uv.sh
python -c "import torch;print('torch',torch.__version__,torch.version.cuda,torch.cuda.is_available())"
echo SETUP_DONE
