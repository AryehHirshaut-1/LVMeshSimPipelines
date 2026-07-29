# **Guide to Creating 3D PV Loops and Calibrating Corresponding 0D Loops:**
1. Choose the Pipeline you want to run (HO - Holzapfel-Ogden, NH - Neo-Hookean).
3. In the 3D folder, run MeshGenerator.py. Set the num_meshes variable to the number of meshes you want. The meshes will download to your local device.
4. In ubuntu, clone the svMultiPhysics repository, and run the lv_bayer.py file in the utilities/fiber_generation folder. You might have to change the file path in the file to your computer's mesh folder (using /mnt/c in the file path). All your created meshes should have fibers automatically added to them.
5. Open Bouchet and upload a clone of the pipeline you are using using the scp command. The meshes should already automatically be saved within the pipeline.
6. Create a virtual environment within the 3D folder of your pipeline. You will need to install the pyvista and numpy packages to your environment to run the VolumeFinder.py file.
7. Run the meshes using the appropriate slurm file using the command sbatch slurmfilename.name. If needed, use the nano slurmfilename.slurm command to open the file, and edit the #SBATCH --array command, changing the range to the number of meshes you generated. The extracted volume files will be automatically extracted and be saved in the VolumeResults Folder, 
10. Use scp again to copy the VolumeResults folder to your local device, copying them into your local copy of the pipeline.
11. Run DatVolumeCombinator.py in the 0D folder, then run calibrationSpherePassiveCSV.py and you're done! A file containing the calibrated gamma (thickness/radius) and n (shape correction factor, or the deviation from a sphere) for all meshes will automatically appear in the 0D folder.
12. (Optional) - Run the 30DResultsComparison.py file to graph your 3D and 0D data. The file outputs four graphs - 3D Gamma vs. 0D Gamma, 3D Height/Radius vs. 0D n, and their corresponding Spearman Coefficient Graphs that plot the ranks of the data points.
