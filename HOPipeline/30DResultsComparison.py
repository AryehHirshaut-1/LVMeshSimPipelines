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

#Separate into varied variables
radiusPoints3 = threed.iloc[0:30]
radiusPoints0 = zerod.iloc[0:30]
heightPoints3 = threed.iloc[30:60]
heightPoints0 = zerod.iloc[30:60]
thicknessPoints3 = threed.iloc[60:90]
thicknessPoints0 = zerod.iloc[60:90]

print(thicknessPoints3)

spearman_coef_gammaR = radiusPoints3["gamma"].corr(radiusPoints0["gamma"])
spearman_coef_gammaH = heightPoints3["gamma"].corr(heightPoints0["gamma"])
spearman_coef_gammaT = thicknessPoints3["gamma"].corr(thicknessPoints0["gamma"])
spearman_coef_nR = radiusPoints3["HR"].corr(radiusPoints0["n"])
spearman_coef_nH = heightPoints3["HR"].corr(heightPoints0["n"])
spearman_coef_nT = thicknessPoints3["HR"].corr(thicknessPoints0["n"])

#Graph Data
fig, axs = plt.subplots(2, 2, figsize = (14,6))

axs[0,0].scatter(radiusPoints3["gamma"], radiusPoints0['gamma'], label="Radius")
axs[0,0].scatter(thicknessPoints3["gamma"], thicknessPoints0['gamma'], label="Thickness")
axs[0,0].set_xlabel("3D Gamma")
axs[0,0].set_ylabel("0D Gamma")
axs[0,0].legend(loc = "upper left")

axs[0,1].scatter(radiusPoints3["gammaRank"], radiusPoints0["gammaRank"], label="Radius")
axs[0,1].scatter(thicknessPoints3["gammaRank"], thicknessPoints0["gammaRank"], label="Thickness")
axs[0,1].set_xlabel("3D Gamma Rank")
axs[0,1].set_ylabel("0D Gamma Rank")
axs[0,1].set_title(f"Spearman Coefficients: Radius = {spearman_coef_gammaR:.2f}, Thickness = {spearman_coef_gammaT:.2f}")
axs[0,1].legend(loc = "upper left")

axs[1,0].scatter(radiusPoints3["Height"]/radiusPoints3["Radius"], radiusPoints0['n'], label="Radius")
axs[1,0].scatter(heightPoints3["Height"]/heightPoints3["Radius"], heightPoints0['n'], label="Height")
axs[1,0].set_xlabel("3D Height/Radius")
axs[1,0].set_ylabel("0D n")
axs[1,0].legend(loc = "upper left")

axs[1,1].scatter(radiusPoints3["HRRank"], radiusPoints0["nRank"], label="Radius")
axs[1,1].scatter(heightPoints3["HRRank"], heightPoints0["nRank"], label="Height")
axs[1,1].set_xlabel("3D Height/Radius Rank")
axs[1,1].set_ylabel("0D n Rank")
axs[1,1].set_title(f"Spearman Coefficients: Radius = {spearman_coef_nR:.2f}, Height = {spearman_coef_nH:.2f}")
axs[1,1].legend(loc = "upper left")

plt.savefig('/users/alanh/Documents/CBLResearch-github/HOPipeline/GammaNCorrelation.png', dpi=300)
plt.show()