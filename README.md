#**Guide to Creating 3D PV Loops and Calibrating Corresponding 0D Loops:**
1. Run MeshGenerator.py. Set the num_meshes variable to the number of meshes you want.
2. Open WSL Terminal, and run AddFibers.py. All your created meshes should have fibers automatically added to them.
3. Open Bouchet and upload run_lv_sweep.slurm. Then use the command sbatch run_lv_sweep.slurm to run the file.
4. Either Copy Results to Local Drive and run VolumeFinder.py locally (not recommended!) or run VolumeFinder within the cluster and download the output CSVs (Very Easy with Claude monitoring the job).
5. Run DatVolumeCombinator.py, then run calibrationSpherePassiveCSV.py and you're done!
