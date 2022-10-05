Bootstrap: docker
From: ghcr.io/truatpasteurdotfr/model-angelo:main

%runscript
# do not use ~/.local python
PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

export TORCH_HOME=/public/model_angelo_weights

eval "$(conda shell.bash hook)"
conda activate model_angelo

#source `which activate` model_angelo
model_angelo "$@"

