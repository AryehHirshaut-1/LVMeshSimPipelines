import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#CSV File Paths
threed = pd.read_csv("HOPipeline/3D/mesh_parameters.csv", header = 0)
zerod = pd.read_csv("HOPipeline/0D/0DOutputs.csv", header = 0)

#Create new Columns for Gamma
threed_gamma = threed["Thickness"]/threed["Radius"]
threed_HR = threed["Height"]/threed["Radius"]


threed["gamma"] = threed["Thickness"]/threed["Radius"]
threed["HR"] = threed["Height"]/threed["Radius"]

#Spearman Rank Correlation Dataframe Columns
threed["gammaRank"] = threed["gamma"].rank()
zerod["gammaRank"] = zerod["gamma"].rank()
threed["HRRank"] = threed["HR"].rank()
zerod["nRank"] = zerod["n"].rank()


#General Plotting
radius3D = threed.iloc[:270]
radius0D = zerod.iloc[:270]
height3D = threed.iloc[270:540]
height0D = zerod.iloc[270:540]
thick3D = threed.iloc[540:]
thick0D = zerod.iloc[540:]

# print(threed)

spearman_coef_gammaR = radius3D["gamma"].corr(radius0D["gamma"])
spearman_coef_gammaT = thick3D["gamma"].corr(thick0D["gamma"])
spearman_coef_nR = radius3D["HR"].corr(radius0D["n"])
spearman_coef_nH = height3D["HR"].corr(height0D["n"])

#Graph Data
fig, axs = plt.subplots(2, 2, figsize = (14,6))

axs[0,0].scatter(radius3D["gamma"], radius0D["gamma"], s=2, label="Radius")
axs[0,0].scatter(thick3D["gamma"], thick0D['gamma'], s=2, label="Thickness", color="orange")
axs[0,0].set_xlabel("3D Gamma")
axs[0,0].set_ylabel("0D Gamma")
axs[0,0].legend()

axs[0,1].scatter(radius3D["gammaRank"], radius0D["gammaRank"], s=2, label="Radius")
axs[0,1].scatter(thick3D["gammaRank"], thick0D["gammaRank"], s=2, label="Thickness", color="orange")
axs[0,1].set_xlabel("3D Gamma Rank")
axs[0,1].set_ylabel("0D Gamma Rank")
axs[0,1].set_title(f"Spearman Coefficient: Radius = {spearman_coef_gammaR:.2f} Thickness = {spearman_coef_gammaT:.2f}")
axs[0,1].legend()

axs[1,0].scatter(radius3D["Height"]/radius3D["Radius"], radius0D['n'], s=2, label="Radius")
axs[1,0].scatter(height3D["Height"]/height3D["Radius"], height0D['n'], s=2, label="Height", color="orange")
axs[1,0].set_xlabel("3D Height/Radius")
axs[1,0].set_ylabel("0D n")
axs[1,0].legend()

axs[1,1].scatter(radius3D["HRRank"], radius0D["nRank"], s=2, label="Radius")
axs[1,1].scatter(height3D["HRRank"], height0D["nRank"], s=2, label="Height", color="orange")
axs[1,1].set_xlabel("3D Height/Radius Rank")
axs[1,1].set_ylabel("0D n Rank")
axs[1,1].set_title(f"Spearman Coefficient: Radius = {spearman_coef_nR:.2f} Height = {spearman_coef_nH:.2f}")
axs[1,1].legend()

plt.savefig('/users/alanh/Documents/CBLResearch-github/HOPipeline/GammaNCorrelationSweep.png', dpi=300)
plt.show()