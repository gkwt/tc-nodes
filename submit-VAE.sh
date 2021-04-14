# !/bin/bash
# SBATCH --gres=gpu:p100:1
# SBATCH --account=def-aspuru
# SBATCH --cpus-per-task=6
# SBATCH --mem=32GB
# SBATCH --time=0-10:00
# SBATCH --output=%N-%j.out

################## If using virtualenv ##################

module load python/3.8

echo "Loading environment."
virtualenv --no-download $SLURM_TMPDIR/csc2516
source $SLURM_TMPDIR/csc2516/bin/activate
pip install --no-index --upgrade pip

# install dgl in directory
cd dgl/python
python setup.py install
cd ..

# install torchsde 
pip install git+https://github.com/google-research/torchsde.git
pip install -r requirements.txt

python run.py

deactivate
