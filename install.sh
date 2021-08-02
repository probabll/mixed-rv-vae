# Setup a virtualenv
virtualenv -p python3 ~/envs/mixedrv
source ~/envs/mixed/bin/activate

# Git clone the code and install requirements
git clone https://github.com/probabll/mixed-rv-vae.git
cd mixed-rv-vae
pip install -r requirements.txt

# Download trained models
wget https://surfdrive.surf.nl/files/index.php/s/r5gx64Nlpoe9GcM/download -O trained_models.tar
tar -xvf trained_models.tar
