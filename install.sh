virtualenv -p python3 ~/envs/mixedrv
source ~/envs/mixed/bin/activate


pip install numpy matplotlib scipy torch torchvision tqdm

mkdir -p ~/workspace/neurips21
cd ~/workspace/neurips21

git clone https://github.com/probabll/dists.pt.git
cd dists.pt
pip install -e .
cd ..

git clone https://github.com/probabll/mixed-rv-vae.git
cd mixed-rv-vae

ls cfg/*
