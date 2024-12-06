#!/bin/bash
#SBATCH -Jnumerosity                            # Job name
#SBATCH -N1                              # Number of nodes and cores per node required
#SBATCH --gres=gpu:H100:1                # Request GPU resources
#SBATCH -t5:00:00                       # Duration of the job (Ex: 14 hours)
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=ojasw@gatech.edu     # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                     # Change to working directory

# conda init                               # Initialize conda
source ~/.bashrc                         # Source the bashrc file
conda activate llama                     # Activate the conda environment
cd /storage/ice-shared/vip-vx4/cs7651    # Change to the directory where the Python script is located

python numerosity.py --stimuli_type random-shape_area
