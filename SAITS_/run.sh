#!/bin/bash
#SBATCH --job-name=TestJobTarek
#SBATCH --output=slurm-%A-out-test-job-TAREK.txt
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_normal_stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --gres=gpu:1


# Aktiviere die Conda-Umgebung
source /home/elounita/miniconda3/bin/activate saits

# Wechsle in das Verzeichnis von Apex
cd /home/elounita/SAITS_
#python saits.py
python saits.py