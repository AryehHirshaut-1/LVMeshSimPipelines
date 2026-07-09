import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#CSV File Paths
threed = pd.read_csv("HOPipeline/3D/mesh_parameters.csv", header = 0)
zerod = pd.read_csv("HOPipeline/0D/0DOutputs.csv", header = 0)

#Create new Columns for Gamma
threed["gamma"] = threed["Thickness"]/threed["Radius"]
threed_height = threed["Height"]
threed_radius = threed["Radius"]

zerod_gamma = zerod["gamma"]
zerod_n = zerod['n']

#Spearman Rank Correlation Dataframe Columns
threed["gammaRank"] = threed["gamma"].rank()
zerod["gammaRank"] = zerod["gamma"].rank()
spearman_coef = threed["gammaRank"].corr(zerod["gammaRank"], method='spearman')


#Graph Data
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14,4))

ax1.scatter(threed["gamma"], zerod["gamma"])
ax1.set_xlabel("3D Gamma")
ax1.set_ylabel("0D Gamma")

ax2.scatter(threed["Height"]/threed["Radius"], zerod["n"])
ax2.set_xlabel("3D Height/Radius")
ax2.set_ylabel("0D n")

ax3.scatter(threed["gammaRank"], zerod["gammaRank"])
m, b = np.polyfit(threed["gammaRank"], zerod["gammaRank"], 1)
ax3.plot(threed["gammaRank"], m * zerod["gammaRank"] + b, color='red')
ax3.set_title(f"Spearman Ranked Scatter Plot (rho = {spearman_coef:.2f})")
ax3.set_xlabel("Rank of 3D Gamma")
ax3.set_ylabel("Rank of 0D Gamma")



plt.show()