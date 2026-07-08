import matplotlib.pyplot as plt
import numpy as np

cases = [0,1,2,3,4]
displacements = [160.03-111.72, 161.57-112.17, 161.87-112.26, 161.94-112.31, 162.10-112.34]


plt.scatter(cases, displacements, marker = 'o', color = 'orange', zorder = 3)
plt.plot(cases, displacements, zorder = 2)
plt.xlabel("Case Number")
plt.ylabel("Displacement Volume (cm^2)")
plt.title("Mesh Convergence (Displacements)")
plt.savefig(r"C:\Users\alanh\Documents\CBLResearch-github\VolumeConvergence\DisplacementVolumes.png", dpi=300)
plt.show()