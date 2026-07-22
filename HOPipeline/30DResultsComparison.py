import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#CSV File Paths
threed = pd.read_csv("HOPipeline/3D/mesh_parameters.csv", header = 0)
zerod = pd.read_csv("HOPipeline/0D/0DOutputs.csv", header = 0)

#Create new Columns for Gamma
threed["gamma"] = threed["Thickness"]/threed["Radius"]
threed["HR"] = threed["Height"]/threed["Radius"]

#Spearman Rank Correlation Dataframe Columns
threed["gammaRank"] = threed["gamma"].rank()
zerod["gammaRank"] = zerod["gamma"].rank()
threed["HRRank"] = threed["HR"].rank()
zerod["nRank"] = zerod["n"].rank()

# spearman_coef_gammaR = radius3D["gamma"].corr(radius0D["gamma"])
spearman_coef_gamma = threed["gamma"].corr(zerod["gamma"])
spearman_coef_n = threed["HR"].corr(zerod["n"])
# spearman_coef_nH = height3D["HR"].corr(height0D["n"])


#---------------------------------------
#-------HR Partitioning By Thickness----
#---------------------------------------
lowbound =.47
highbound=.53

lowerthick3 = threed[threed["Thickness"] <= lowbound]
mediumthick3 = threed[(threed["Thickness"] > lowbound) & (threed["Thickness"] <= highbound)]
highthick3 = threed[threed["Thickness"] > highbound]
lowerthick0 = zerod.loc[lowerthick3.index]
mediumthick0 = zerod.loc[mediumthick3.index]
highthick0 = zerod.loc[highthick3.index]

conditions = [
    threed["Thickness"] <= lowbound,
    (threed["Thickness"] > lowbound) & (threed["Thickness"] <= highbound),
    threed["Thickness"] > highbound
]
choices = ["Low", "Middle", "High"]

threed["ThicknessGroup"] = np.select(conditions, choices, default="unknown")

spearman_coef_HR_lower = lowerthick3["HR"].corr(lowerthick0["n"])
spearman_coef_HR_medium = mediumthick3["HR"].corr(mediumthick0["n"])
spearman_coef_HR_high = highthick3["HR"].corr(highthick0["n"])










#---------------------------------------
#-------------Graphing Data-------------
#---------------------------------------
fig, axs = plt.subplots(2, 2, figsize = (14,6))

color_map = {"Low": "tab:blue", "Middle": "tab:orange", "High": "tab:red"}
colors = threed["ThicknessGroup"].map(color_map)

axs[0,0].scatter(threed["gamma"], zerod['gamma'], s=10)
axs[0,0].set_xlabel("3D Gamma")
axs[0,0].set_ylabel("0D Gamma")

axs[0,1].scatter(threed["HR"], zerod["n"], s=10, c=colors)
axs[0,1].set_xlabel("3D Height/Radius")
axs[0,1].set_ylabel("0D n")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=label)
           for label, c in color_map.items()]
axs[0,1].legend(handles=handles, title="Thickness Group")

axs[1,0].scatter(threed["gammaRank"], zerod["gammaRank"], s=10)
axs[1,0].set_xlabel("3D Gamma Rank")
axs[1,0].set_ylabel("0D Gamma Rank")
axs[1,0].set_title(f"Spearman Coefficient: {spearman_coef_gamma:.2f}")

#Spearman Correlation Example
axs[1,1].scatter(threed["HRRank"], zerod["nRank"], s=10, c=colors)
axs[1,1].set_xlabel("3D Height/Radius Rank")
axs[1,1].set_ylabel("0D n Rank")
axs[1,1].set_title(f"Spearman Coefficients")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8,
           label=f"{label.capitalize()}: {threed.loc[threed['ThicknessGroup']==label, 'HR'].corr(zerod.loc[threed['ThicknessGroup']==label, 'n'], method='spearman'):.2f}")
           for label, c in color_map.items()]
axs[1,1].legend(handles=handles, title="Thickness Group", loc="upper left")

fig.suptitle('3D-0D Calibration Parameter Fit with Pig Heart Data', fontsize=16, fontweight='bold')

fig.tight_layout()
plt.savefig('/users/alanh/Documents/CBLResearch-github/HOPipeline/GammaNPigCorrelationSweep.png', dpi=300)
#plt.show()