#!/usr/bin/env python
# -*-coding:utf-8 -*
# 
# 
# Only works in WSL Terminal - Requires a built svMultiPhysics to run!
"""Main script for generating biventricular fibers using the Bayer method.

This module implements fiber generation for biventricular heart models using
the Laplace-Dirichlet rule-based method described in:
Bayer et al. 2012, "A Novel Rule-Based Algorithm for Assigning Myocardial 
Fiber Orientation to Computational Heart Models"
https://doi.org/10.1007/s10439-012-0593-5

The script supports command-line arguments for customization of mesh paths,
output directories, and solver executables.
"""

import argparse
import os
import pyvista as pv
from src.LaplaceSolver import LaplaceSolver
from src.FibGen import FibGenBayerLV
from src.SurfaceNames import SurfaceName
from src.surface_utils import generate_epi_apex
from time import time
from pathlib import Path


if __name__ == "__main__":

    folder_path = Path("/mnt/c/users/alanh/Documents/CBLResearch-github/lv_sim_cases/VolConvergence")
    folder_count = sum(1 for entry in os.scandir(folder_path) if entry.is_dir())
    print(folder_count)
    
    for file_num in range(0, folder_count):
        run_flag = True
        svmultiphysics_exec = "svmultiphysics"

        mesh_path = f"/mnt/c/users/alanh/Documents/CBLResearch-github/lv_sim_cases/VolConvergence/case_{file_num}/mesh_{file_num}_volume.vtu"
        outdir = f"/mnt/c/users/alanh/Documents/CBLResearch-github/lv_sim_cases/VolConvergence/case_{file_num}/mesh_{file_num}_fibers"
        surfaces_dir = f"/mnt/c/users/alanh/Documents/CBLResearch-github/lv_sim_cases/VolConvergence/case_{file_num}"
                        
        # Parameters for the Bayer et al. method https://doi.org/10.1007/s10439-012-0593-5. 
        params = {
            "ALFA_END": 60.0,
            "ALFA_EPI": -60.0,
            "BETA_END": -20.0,
            "BETA_EPI": 20.0,
        }


        ###########################################################
        ############  FIBER GENERATION  ###########################
        ###########################################################

        # Optional CLI overrides
        parser = argparse.ArgumentParser(description="Generate fibers using the Bayer method.")
        parser.add_argument("--svmultiphysics-exec", default=svmultiphysics_exec, help="svMultiPhysics executable/command (default: %(default)s)")
        parser.add_argument("--mesh-path", default=mesh_path, help="Path to the volumetric mesh .vtu (default: %(default)s)")
        parser.add_argument(
            "--surfaces-dir",
            default=surfaces_dir,
            help="Directory containing mesh surfaces; default: <parent of mesh_path>/mesh-surfaces",
        )
        parser.add_argument("--outdir", default=outdir, help="Output directory (default: %(default)s)")
        args = parser.parse_args()

        svmultiphysics_exec = args.svmultiphysics_exec
        if not svmultiphysics_exec.endswith(" "):
            svmultiphysics_exec = svmultiphysics_exec + " "

        mesh_path = args.mesh_path
        outdir = args.outdir

        if args.surfaces_dir is None:
            surfaces_dir = os.path.join(os.path.dirname(mesh_path), "mesh-surfaces")
        else:
            surfaces_dir = os.path.abspath(args.surfaces_dir)

        # Make sure the paths are full paths
        mesh_path = os.path.abspath(mesh_path)
        outdir = os.path.abspath(outdir)
        surfaces_dir = os.path.abspath(surfaces_dir)

        # Define surface paths
        surface_paths = {SurfaceName.EPICARDIUM: f'{surfaces_dir}/mesh_{file_num}_epi.vtp',
                        SurfaceName.EPICARDIUM_APEX: f'{surfaces_dir}/mesh_{file_num}_epi_apex.vtp',
                        SurfaceName.BASE: f'{surfaces_dir}/mesh_{file_num}_base.vtp',
                        SurfaceName.ENDOCARDIUM_LV: f'{surfaces_dir}/mesh_{file_num}_endo.vtp'
                        }
        
        # Create output directory if needed
        os.makedirs(outdir, exist_ok=True)
        
        # Check if the EPICARDIUM_APEX surface exists; if not create it
        start = time()
        if not os.path.exists(surface_paths[SurfaceName.EPICARDIUM_APEX]):
            print("Generating EPICARDIUM_APEX surface...")
            generate_epi_apex(surface_paths)
            
        # Initialize Laplace solver
        solver = LaplaceSolver(mesh_path, surface_paths, svmultiphysics_exec)

        # Run the Laplace solver
        if run_flag:
            print("Running Laplace solver...")
            laplace_results_file = solver.run("bayer_lv", outdir, file_num)
        else:
            laplace_results_file = os.path.join(outdir, 'laplace_results.vtu')

        # Initialize fiber generator
        print("\nGenerating fibers using Bayer method...")
        fib_gen = FibGenBayerLV()

        # Load Laplace results
        fib_gen.load_laplace_results(laplace_results_file)

        # Generate fiber directions
        F, S, T = fib_gen.generate_fibers(params)
        print(f"generate fibers (Bayer method) elapsed time: {time() - start:.3f} s")
        
        # Write fibers to output directory
        fib_gen.write_fibers(outdir)

        # Save the result mesh
        result_mesh_path = os.path.join(outdir, "fiber_mesh.vtu")
        fib_gen.mesh.save(result_mesh_path)
        os.remove(os.path.join(outdir, "svFibers_BiV.xml"))
        os.remove(os.path.join(outdir, "histor.dat"))
        print(f"\nResults saved to: {result_mesh_path}")