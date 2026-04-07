import os
from ase.io import read, write

f=open('submit.job','w')
f.write("#!/bin/bash\n")
f.write("#SBATCH --job-name=act_learn\n")
f.write("#SBATCH --partition=qgpu\n")
f.write("#SBATCH --mem=5G\n")
f.write("#SBATCH --ntasks=1\n")
f.write("#SBATCH --time=12:00:00\n")
f.write('echo "========= Job started at `date` =========="\n')
f.write(f'command="python active_learning.py"\n')
f.write("$command\n")
f.write('echo "========= Job ended at `date` =========="\n')
f.close()
os.system("sbatch submit.job")