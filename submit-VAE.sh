#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=rrg-aspuru
#SBATCH --time=0-10:00:00
#SBATCH --output=%N-%j-VAE.out

################# If running on MIST #####################


module load cuda/10.2.89
module load anaconda3

echo "Loading hgryff environment."
source activate hgryff

echo ""
echo "Starting run."
echo $(date '+%d/%m/%Y %H:%M:%S')
echo ""
python run_vae_conv_fcnn.py
echo "Run complete."
echo $(date '+%d/%m/%Y %H:%M:%S')

conda deactivate


# # !/bin/bash
# # SBATCH --gres=gpu:p100:1
# # SBATCH --account=def-aspuru
# # SBATCH --cpus-per-task=6
# # SBATCH --mem=32GB
# # SBATCH --time=1-12:00
# # SBATCH --output=%N-%j.out

################### If using virtualenv ##################

# module load python/3.6.3
# module load nixpkgs/16.09  gcc/7.3.0
# module load rdkit/2019.03.4
# module load cudacore/.10.1.243 cudnn/7.6.5

# echo "Loading environment."
# virtualenv --no-download $SLURM_TMPDIR/hgryff
# source $SLURM_TMPDIR/hgryff/bin/activate
# pip install --no-index --upgrade pip
# pip install -r requirements.txt

# echo "Running VAE training code on QM9."
# python VAE_selfies_nonfull.py
# echo "Run complete."

# deactivate
