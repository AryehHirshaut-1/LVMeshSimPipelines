import matplotlib.pyplot as plt

cases = [0,1,2,3]
displacements = [160.03-111.72, 161.57-112.17, 161.87-112.26, 161.94-112.31]

plt.plot(cases, displacements)
plt.xlabel("Case Number")
plt.ylabel("Displacement Volume (cm^2)")
plt.title("VolumeConvergence")
plt.show()