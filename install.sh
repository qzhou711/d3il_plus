#!/bin/bash
set -e

############ GENERAL ENV SETUP ############
echo New Environment Name:
read envname

echo Creating new conda environment $envname
conda create -n "$envname" python=3.10.8 -y -q

eval "$(conda shell.bash hook)"
conda activate "$envname"

echo
echo Activating $envname
if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
then
    echo Failed to activate conda environment.
    exit 1
fi

### Set Channel vars
conda config --add channels conda-forge
conda config --set channel_priority strict




############ PYTHON DEPENDENCIES ############
echo Installing Python dependencies...

# gym 0.21.0 has legacy package metadata that newer pip/setuptools reject.
python -m pip install --upgrade "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install "gym==0.21.0" --no-build-isolation
python -m pip install -r requirements.txt

# ############ PYTHON ############
# echo Install mamba
# conda install mamba -c conda-forge -y -q


# ############ REQUIRED DEPENDENCIES (PYBULLET) ############
# echo Installing dependencies...

# mamba install -c conda-forge pytorch==1.13.0 torchvision==0.14.0

# mamba install -c conda-forge pybullet pyyaml scipy opencv pinocchio matplotlib gin-config gym==0.21.0 -y -q

# # Open3D for PointClouds and its dependencies. Why does it not install them directly?
# mamba install -c conda-forge scikit-learn addict pandas plyfile tqdm -y -q
# mamba install -c open3d-admin open3d -y -q

# pip install einops
# pip install hydra-core==1.1.1
# pip install wandb

# # Robomimic
# pip install termcolor

# # ACT
# pip install ipython

# # BESO
# pip install torchsde torchdiffeq

# ############ MUJOCO BETA SUPPORT INSTALLATION ############
# mamba install -c conda-forge imageio -y -q
# pip install mujoco==2.3.2

# ############ INSTALL D3il-Sim & FINALIZE ############
# echo
# echo Installing D3il-Sim Package
# cd environments/d3il && pip install -e .

exit 0
