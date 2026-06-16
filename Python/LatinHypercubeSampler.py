import numpy as np
from scipy.stats import norm
from scipy.stats import qmc  # Quasi-Monte Carlo submodule

#Change these ranges to reflect the physiological variability of the LV geometry
ranges = {
    'radius': (2.1, 2.95),     # Inner radius
    'height': (5, 12),     # Distance from apex to base
    'thickness': (0.24, 0.42)    # Myocardial wall thickness
}

num_samples = 100
num_vars = len(ranges)

dist_params = {}
for var, (vmin, vmax) in ranges.items():
    mu = (vmin + vmax) / 2.0
    sigma = (vmax - vmin) / (2 * 1.96)
    dist_params[var] = (mu, sigma)


sampler = qmc.LatinHypercube(d=num_vars)
lhs_samples = sampler.random(n=num_samples) 

lv_models = np.zeros((num_samples, num_vars))
var_names = list(ranges.keys())

for i, var in enumerate(var_names):
    mu, sigma = dist_params[var]
    # Transform using the percent point function (inverse CDF)
    raw_samples = norm.ppf(lhs_samples[:, i], loc=mu, scale=sigma)
    
    # ADDITION: Cap the values exactly at the defined range bounds
    vmin, vmax = ranges[var]
    lv_models[:, i] = np.clip(raw_samples, a_min=vmin, a_max=vmax)

lv_models = np.round(lv_models, decimals=3)

print(f"Successfully generated {num_samples} Left Ventricle configurations.\n")
print(f"{'Model #':<10}{'Radius':<12}{'Height':<12}{'Thickness':<12}")
print("-" * 48)
for i in range(min(10, num_samples)): 
    print(f"{i+1:<10}{lv_models[i,0]:<12.2f}{lv_models[i,1]:<12.2f}{lv_models[i,2]:<12.2f}")
print(lv_models)


#Plot Generated Parameters
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Left Ventricle Parameter Analysis (LHS + Gaussian Distribution)', fontsize=16, fontweight='bold')

model_numbers = np.arange(1, num_samples + 1)

for i, var in enumerate(var_names):
    ax_scatter = axes[0, i]
    ax_scatter.scatter(model_numbers, lv_models[:, i], color='teal', alpha=0.7, edgecolors='k', s=25)
    
    ax_scatter.set_title(f'{var.capitalize()} vs. Model Number', fontsize=12, fontweight='semibold')
    ax_scatter.set_xlabel('Model Number', fontsize=10)
    ax_scatter.set_ylabel(f'{var.capitalize()} Value', fontsize=10)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    
    # --- ROW 2: Histograms (Frequency Distribution) ---
    ax_hist = axes[1, i]
    ax_hist.hist(lv_models[:, i], bins=15, color='darkslategray', edgecolor='black', alpha=0.7)
    
    ax_hist.set_title(f'{var.capitalize()} Frequency', fontsize=12, fontweight='semibold')
    ax_hist.set_xlabel(f'{var.capitalize()} Value', fontsize=10)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.grid(True, linestyle='--', alpha=0.5)


# Adjust layout so labels don't overlap
plt.tight_layout()
plt.savefig('/users/alanh/Documents/CBLResearch/Python/lv_model_parameters.png', dpi=300)  # Save the figure as a high-resolution PNG
plt.show()