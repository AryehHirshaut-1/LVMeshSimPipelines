import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares

# ============================================================
# USER SETTINGS
# ============================================================

# Select material model: "NH" or "HO"
MATERIAL_MODEL = "HO"

# Input file: a CSV with exactly 3 rows --
#   row 1 = time
#   row 2 = pressure
#   row 3 = volume

# Numerical options
NORMALIZE_COST = True # divide residuals by sqrt(N)*max(Pout) -> dimensionless RMS relative cost
MAKE_LANDSCAPE_PLOT = False
PLOT_RESIDUAL_TERMS = False

# Toggle to compare the fitted model against the uploaded/input data.
# This only makes plots; it does not print metrics or export files.
PLOT_FITTED_MODEL_VS_DATA = False

# Which points to fit. For an imaging-derived passive inflation curve,
# it is often better to fit only the inflation limb.
# Options: "all", "inflation", "deflation"
FIT_BRANCH = "all"

# Toggle to also estimate material parameters
OPTIMIZE_MATERIAL_PARAMS = False

# Include eta in the material-parameter optimization for the selected model.
# IMPORTANT: eta is only meaningful if the time scale and dV/dt are reliable.
# If OPTIMIZE_MATERIAL_PARAMS=False, this toggle has no effect.
OPTIMIZE_ETA = False

# Landscape plot ranges in (x,y) = (log(gamma), log(n))
LAND_X_RANGE = (np.log(1e-6), np.log(2.0))
LAND_Y_RANGE = (np.log(0.2), np.log(5.0))
LAND_NX = 120
LAND_NY = 120

# ============================================================
# COMMON MODEL / ACTIVATION PARAMETERS
# ============================================================

rho = 1.0

# Activation model parameters (passive by default)
sigma_max = 0.0
alpha_max = 0.0
alpha_min = 0.0
tsys = 0.0
tdias = 0.0
steepness = 1e-9
T_cardiac = 1.6119

# ============================================================
# MATERIAL PARAMETERS
# ============================================================

# Neohookean-like spherical model:
# W1 and W2 are interpreted as dWe_dJ1 and dWe_dJ2
NH_PARAMS = {
    "W1": 1.69e5,
    "W2": 0.0,
    "eta": 0.0,
}

# Holzapfel-Ogden model
HO_PARAMS = {
    "a": 14720.0,
    "b": 33.39,
    "a4f": 0,
    "b4f": 0,
    "a4s": 0,
    "b4s": 0,
    "eta": 0,
}

# Initial guesses for geometry parameters
# theta[0] = log(gamma) where gamma = thick0/radius0
# theta[1] = log(n)
INITIAL_GAMMA_GUESS = 0.13
INITIAL_N_GUESS = 1.0

# Initial guesses for material parameters when OPTIMIZE_MATERIAL_PARAMS = True
# NH: W2 is always forced to 0.
# eta must be strictly positive if optimized in log-space; use a small nonzero
# guess even when the default/base eta is zero.
NH_OPT_INIT = {
    "W1": 0.2*NH_PARAMS["W1"],
    "eta": max(NH_PARAMS["eta"], 1.0),
}

HO_OPT_INIT = {
    "a": 0.5 * HO_PARAMS["a"],
    "b": 0.5 * HO_PARAMS["b"],
    "a4f": 0.5 * HO_PARAMS["a4f"],
    "b4f": 0.5 * HO_PARAMS["b4f"],
    "a4s": 0.5 * HO_PARAMS["a4s"],
    "b4s": 0.5 * HO_PARAMS["b4s"],
    "eta": max(HO_PARAMS["eta"], 1.0),
}

# ============================================================
# HELPERS
# ============================================================

def safe_exp(z, zmax=80.0):
    """Prevent floating-point overflow in exponentials."""
    return np.exp(np.clip(z, -zmax, zmax))


def compute_tau(time):
    """
    Discrete active stress update.
    Passive default => sigma_max = alpha_max = alpha_min = 0 => tau = 0.
    """
    N = len(time)
    tau_vec = np.zeros(N)
    tau_prev = 0.0

    for k in range(N):
        dt = (time[1] - time[0]) if k == 0 else (time[k] - time[k - 1])

        t_in_cycle = np.mod(time[k], T_cardiac)
        S_plus = 0.5 * (1.0 + np.tanh((t_in_cycle - tsys) / steepness))
        S_minus = 0.5 * (1.0 - np.tanh((t_in_cycle - tdias) / steepness))
        f = S_plus * S_minus

        act_t = alpha_max * f + alpha_min * (1.0 - f)
        act = abs(act_t)
        act_plus = max(act_t, 0.0)

        tau = (tau_prev + dt * sigma_max * act_plus) / (1.0 + dt * act)
        tau_vec[k] = tau
        tau_prev = tau

    return tau_vec


def load_time_pressure_volume_from_csv(csv_path):
    """
    Load a CSV where:
      column 1 = time
      column 2 = pressure
      column 3 = volume

    One row per sample; no header row expected. Returns
    (time, pressure, volume) as float arrays.
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if len(row) > 0]

    if len(rows) == 0:
        raise ValueError(f"No data found in '{csv_path}'.")

    time_list, pressure_list, volume_list = [], [], []

    for i, row in enumerate(rows):
        if len(row) < 3:
            raise ValueError(
                f"Row {i} in '{csv_path}' has fewer than 3 columns: {row}"
            )
        time_list.append(float(row[0]))
        pressure_list.append(float(row[1]))
        volume_list.append(float(row[2]))

    time = np.array(time_list, dtype=float)
    pressure = np.array(pressure_list, dtype=float)
    volume = np.array(volume_list, dtype=float)

    return time, pressure, volume


def get_optimized_material_keys(model):
    if not OPTIMIZE_MATERIAL_PARAMS:
        return []

    if model == "NH":
        # W2 must stay at zero. eta can optionally be estimated.
        keys = ["W1"]
    elif model == "HO":
        # Keep the existing material-property estimation behavior.
        # Fiber terms remain fixed unless you add them here.
        keys = ["a", "b"]
    else:
        raise ValueError(f"Unknown material model '{model}'. Use 'NH' or 'HO'.")

    if OPTIMIZE_ETA:
        keys.append("eta")

    return keys

def get_material_init_guess(model):
    if model == "NH":
        return NH_OPT_INIT
    if model == "HO":
        return HO_OPT_INIT
    raise ValueError(f"Unknown material model '{model}'. Use 'NH' or 'HO'.")


def update_material_params_from_theta(base_params, theta, model):
    """
    Read optimized material parameters from theta and return a fresh params dict.

    Parameterization:
      - geometry uses theta[0:2] = [log(gamma), log(n)]
      - material parameters use log(parameter) so positivity is enforced
      - NH keeps W2 = 0.0 always
    """
    params = dict(base_params)
    offset = 2

    for key in get_optimized_material_keys(model):
        params[key] = float(np.exp(theta[offset]))
        offset += 1

    if model == "NH":
        params["W2"] = 0.0

    return params


def theta_to_named_dict(theta, model):
    named = {
        "log(gamma)": float(theta[0]),
        "log(n)": float(theta[1]),
    }

    offset = 2
    for key in get_optimized_material_keys(model):
        named[f"log({key})"] = float(theta[offset])
        offset += 1

    return named


# ============================================================
# MATERIAL RESPONSE
# ============================================================

def material_response(C, Cdot, model, params):
    """
    Return constitutive ingredients for the spherical stress:
        dWe_dJ1, dWe_dJ2, dWe_dJ4, visc

    C    = (R / radius0)^2
    Cdot = d/dt[(R / radius0)^2]
    """
    if model == "NH":
        dWe_dJ1 = params["W1"]
        dWe_dJ2 = params["W2"]
        dWe_dJ4 = np.zeros_like(C)

        # Keep the same viscous term used in your NH script
        visc = 2.0 * params["eta"] * Cdot * (1.0 - 2.0 * C**-6)

    elif model == "HO":
        J1 = (1.0 / C**2) + 2.0 * C

        dWe_dJ1 = 0.5 * params["a"] * safe_exp(params["b"] * (J1 - 3.0))
        dWe_dJ2 = np.zeros_like(C)

        macaulay = np.maximum(C - 1.0, 0.0)
        dWe_dJ4 = (
            params["a4f"] * macaulay * safe_exp(params["b4f"] * macaulay**2)
            + params["a4s"] * macaulay * safe_exp(params["b4s"] * macaulay**2)
        )

        # Keep the same viscous term used in your HO script
        visc = params["eta"] * Cdot * (1.0 + 2.0 * C**-6)

    else:
        raise ValueError(f"Unknown material model '{model}'. Use 'NH' or 'HO'.")

    return dWe_dJ1, dWe_dJ2, dWe_dJ4, visc


def spherical_stress(C, Cdot, tau_vec, model, params):
    """
    Common spherical stress structure for both material models.
    """
    dWe_dJ1, dWe_dJ2, dWe_dJ4, visc = material_response(C, Cdot, model, params)

    return (
        tau_vec
        + 4.0 * (1.0 - C**-3) * (dWe_dJ1 + C * dWe_dJ2)
        + 2.0 * dWe_dJ4
        + visc
    )




# ============================================================
# LOAD DATA
# ============================================================

CSV_PATH = "0DModeling/SampleInputData/time_pressure_volume.csv"

time, Pout, volume = load_time_pressure_volume_from_csv(CSV_PATH)

# No separate dV/dt source in the CSV format, so always differentiate numerically.
dvolume = np.gradient(volume, time)

print("Loaded:")
print(f"  MATERIAL_MODEL = {MATERIAL_MODEL}")
print(f"  CSV_PATH       = {CSV_PATH}")
print(f"  N points       = {len(time)}")
print(f"  time range     = [{time[0]:.6f}, {time[-1]:.6f}]")
print(f"  Pout range     = [{Pout.min():.6e}, {Pout.max():.6e}]")
print(f"  volume range   = [{volume.min():.6e}, {volume.max():.6e}]")
print(f"  dV/dt range    = [{dvolume.min():.6e}, {dvolume.max():.6e}]")
print()

# ============================================================
# PRECOMPUTE KINEMATICS INPUTS
# ============================================================

tau_vec = compute_tau(time)

V_safe = np.maximum(volume, 1e-12)
V0 = V_safe[0]
Vratio = V_safe / V0

# Initial guess in reparameterized variables:
# theta[0] = log(gamma)
# theta[1] = log(n)
g0 = np.log(INITIAL_GAMMA_GUESS)
n0 = np.log(INITIAL_N_GUESS)

norm_factor = 1.0

if NORMALIZE_COST:
    norm_factor = 1.0 / (np.sqrt(len(time)) * Pout.max())
    print(f"Cost normalization: 1/(sqrt(N)*max(Pout)) = {norm_factor:.4e}")
    print(f"  Normalized cost ≈ RMS residual / max(Pout)  (dimensionless)\n")

# Material parameter selection
if MATERIAL_MODEL == "NH":
    material_params = dict(NH_PARAMS)
elif MATERIAL_MODEL == "HO":
    material_params = dict(HO_PARAMS)
else:
    raise ValueError("MATERIAL_MODEL must be 'NH' or 'HO'.")

# ============================================================
# FIT MASK
# ============================================================

if FIT_BRANCH == "all":
    fit_mask = np.ones_like(time, dtype=bool)
elif FIT_BRANCH == "inflation":
    #fit_mask = dvolume > 0.0
    imax = np.argmax(Pout)
    fit_mask = np.arange(len(Pout)) <= imax
elif FIT_BRANCH == "deflation":
    fit_mask = dvolume < 0.0
else:
    raise ValueError("FIT_BRANCH must be 'all', 'inflation', or 'deflation'.")

if np.count_nonzero(fit_mask) < 3:
    raise ValueError(f"FIT_BRANCH='{FIT_BRANCH}' leaves too few points to fit.")

print(f"  FIT_BRANCH     = {FIT_BRANCH} ({np.count_nonzero(fit_mask)}/{len(fit_mask)} points)")
print()

# ============================================================
# RESIDUALS
# ============================================================

def residual_components(theta):
    """
    Return each term of the force-balance residual separately.

    theta contains:
      theta[0] = log(gamma)
      theta[1] = log(n)
      optionally additional log(material parameters), including eta when OPTIMIZE_ETA=True

    with
      gamma = thick0 / radius0
      V(t)/V(0) = (R(t)/radius0)^(3n)
    """
    log_gamma, log_n = theta[:2]

    gamma = np.exp(log_gamma)
    n = np.exp(log_n)

    current_material_params = update_material_params_from_theta(
        material_params, theta, MATERIAL_MODEL
    )

    # Generalized kinematics:
    # lam = R/radius0 = (V/V0)^(1/(3n))
    lam = Vratio ** (1.0 / (3.0 * n))
    C = lam**2

    # dlam/dt = lam * (1/(3n)) * (dV/dt)/V
    lam_dot = lam * (1.0 / (3.0 * n)) * (dvolume / V_safe)
    Cdot = 2.0 * lam * lam_dot

    stress = spherical_stress(C, Cdot, tau_vec, MATERIAL_MODEL, current_material_params)

    # Inertia dropped
    inert = np.zeros_like(time)
    traction = gamma * lam * stress
    press = Pout * C

    res = inert + traction - press
    return inert, traction, press, res


def residual_theta(theta):
    """
    Normalized force-balance residual used by least_squares.

    theta contains:
      theta[0] = log(gamma)
      theta[1] = log(n)
      optionally additional log(material parameters)
    """
    inert, traction, press, res = residual_components(theta)
    r_main = (norm_factor * res)[fit_mask]

    return r_main


def plot_residual_terms(theta, title="Residual term contributions"):
    inert, traction, press, res = residual_components(theta)

    inert_p = norm_factor * inert
    tract_p = norm_factor * traction
    press_p = norm_factor * press
    res_p = norm_factor * res

    plt.figure(figsize=(10, 6))
    plt.plot(time, inert_p, label="inertia: dropped (=0)")
    plt.plot(time, tract_p, label="traction: gamma*(R/radius0)*stress")
    plt.plot(time, -press_p, label="pressure term: -Pout*C")
    plt.plot(time, res_p, "--", linewidth=2, label="total residual r(t)")
    plt.xlabel("time (s)")
    plt.ylabel("term value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    def rms(x):
        return np.sqrt(np.mean(x**2))

    print(title)
    print(f"  inertia   RMS: {rms(inert):.3e}")
    print(f"  traction  RMS: {rms(traction):.3e}")
    print(f"  pressure  RMS: {rms(press):.3e}")
    print(f"  residual  RMS: {rms(res):.3e}")
    print()

def residual_theta_logged(theta):
    """
    Wrapper that logs the optimization path.
    """
    global _last_theta

    theta = np.asarray(theta, dtype=float)

    if _last_theta is None or np.linalg.norm(theta - _last_theta) > 1e-12:
        r = residual_theta(theta)
        trajectory.append(theta.copy())
        cost_traj.append(float(np.linalg.norm(r)))
        _last_theta = theta.copy()

    return residual_theta(theta)

# ============================================================
# OPTIMIZATION
# ============================================================

theta_init_list = [g0, n0]
for key in get_optimized_material_keys(MATERIAL_MODEL):
    guess_val = get_material_init_guess(MATERIAL_MODEL)[key]
    if guess_val <= 0.0:
        raise ValueError(f"Initial guess for '{key}' must be > 0 when optimizing in log-space.")
    theta_init_list.append(np.log(guess_val))

theta_init = np.array(theta_init_list, dtype=float)

trajectory = []
cost_traj = []
_last_theta = None

if PLOT_RESIDUAL_TERMS:
    plot_residual_terms(theta_init, title="Initial guess term contributions")





sol = least_squares(
    residual_theta_logged,
    theta_init,
    method="lm",
    jac="2-point",
    max_nfev=2000,
)

log_gamma_hat, log_n_hat = sol.x[:2]
gamma_hat = float(np.exp(log_gamma_hat))
n_hat = float(np.exp(log_n_hat))

opt_material_params = update_material_params_from_theta(
    material_params, sol.x, MATERIAL_MODEL
)

if PLOT_RESIDUAL_TERMS:
    plot_residual_terms(sol.x, title="Optimized term contributions")

print("=== Optimization result ===")
print(f"MATERIAL_MODEL = {MATERIAL_MODEL}")
print(f"OPTIMIZE_MATERIAL_PARAMS = {OPTIMIZE_MATERIAL_PARAMS}")
print(f"OPTIMIZE_ETA = {OPTIMIZE_ETA}")
print(f"theta* = {theta_to_named_dict(sol.x, MATERIAL_MODEL)}")
print(f"gamma* = {gamma_hat:.9f}")
print(f"n*     = {n_hat:.9f}")

if OPTIMIZE_MATERIAL_PARAMS:
    print("optimized material parameters:")
    if MATERIAL_MODEL == "NH":
        print(f"  W1*  = {opt_material_params['W1']:.9e}")
        print(f"  W2*  = {opt_material_params['W2']:.9e}  (fixed)")
        print(f"  eta* = {opt_material_params['eta']:.9e}")
        print(f"  gamma*W1*  = {gamma_hat * opt_material_params['W1']:.9e}")
        print(f"  gamma*eta* = {gamma_hat * opt_material_params['eta']:.9e}")
    elif MATERIAL_MODEL == "HO":
        for key in ["a", "b", "a4f", "b4f", "a4s", "b4s", "eta"]:
            suffix = "" if key in get_optimized_material_keys(MATERIAL_MODEL) else "  (fixed)"
            print(f"  {key:>3s}* = {opt_material_params[key]:.9e}{suffix}")
        print(f"  gamma*a*   = {gamma_hat * opt_material_params['a']:.9e}")
        print(f"  gamma*eta* = {gamma_hat * opt_material_params['eta']:.9e}")

print(f"final cost = {sol.cost:.6e}  (0.5*||r||^2)")
print(f"||r||_2     = {np.linalg.norm(sol.fun):.6e}")
print(f"nfev        = {sol.nfev}")
print(f"status      = {sol.status}  ({sol.message})")
print()

# Ensure trajectory has start/end
if len(trajectory) == 0:
    trajectory = [theta_init.copy(), sol.x.copy()]
    cost_traj = [
        float(np.linalg.norm(residual_theta(theta_init))),
        float(np.linalg.norm(sol.fun)),
    ]
else:
    if np.linalg.norm(trajectory[0] - theta_init) > 0:
        trajectory.insert(0, theta_init.copy())
        cost_traj.insert(0, float(np.linalg.norm(residual_theta(theta_init))))
    if np.linalg.norm(trajectory[-1] - sol.x) > 0:
        trajectory.append(sol.x.copy())
        cost_traj.append(float(np.linalg.norm(sol.fun)))

traj = np.vstack(trajectory)

# ============================================================
# OPTIONAL LANDSCAPE PLOT
# ============================================================

# This 2D landscape is only meaningful when the optimization variables are
# restricted to [log(gamma), log(n)].
if MAKE_LANDSCAPE_PLOT and OPTIMIZE_MATERIAL_PARAMS:
    print("Skipping landscape plot because OPTIMIZE_MATERIAL_PARAMS=True adds")
    print("extra optimization dimensions beyond [log(gamma), log(n)].")
    print()

# ============================================================
# OPTIONAL: LANDSCAPE PLOT IN (log(gamma), log(n))
# ============================================================
if MAKE_LANDSCAPE_PLOT and not OPTIMIZE_MATERIAL_PARAMS:
    xs = np.linspace(LAND_X_RANGE[0], LAND_X_RANGE[1], LAND_NX)  # log(gamma)
    ys = np.linspace(LAND_Y_RANGE[0], LAND_Y_RANGE[1], LAND_NY)  # log(n)

    Z = np.zeros((LAND_NY, LAND_NX), dtype=float)

    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            rvec = residual_theta([xx, yy])
            Z[j, i] = np.linalg.norm(rvec)

    Zpos = Z[Z > 0]
    vmin = Zpos.min()
    vmax = Zpos.max()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap="inferno_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[xs[0], xs[-1], ys[0], ys[-1]]
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Cost = $\|r\|_2$", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    levels = np.geomspace(np.min(Z[Z > 0]), np.max(Z), 12)
    plt.contour(xs, ys, Z, levels=levels, colors="white", linewidths=1.0, alpha=0.8)

    # LM trajectory
    plt.plot(
        traj[:, 0], traj[:, 1],
        "-o", markersize=3, linewidth=2, color="black",
        label="LM trajectory"
    )

    # start / end
    plt.plot(
        [theta_init[0]], [theta_init[1]],
        "o", markersize=10, color="lime", label="start"
    )
    plt.plot(
        [log_gamma_hat], [log_n_hat],
        marker="x", markersize=12, linewidth=3, color="lime", label="end"
    )

    plt.plot([log_gamma_hat], [log_n_hat], marker="x", markersize=10, linewidth=3)

    plt.xlabel("log($\\gamma$)", fontsize=20)
    plt.ylabel("log($n$)", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================
# OPTIONAL FITTED MODEL VS INPUT DATA PLOTS

# FORWARD EVALUATION IS VOLUME-DRIVEN
# ============================================================

def compute_fitted_model_for_plot(theta):
    """Compute pointwise fitted pressure values for plotting only."""
    inert, traction, press, res = residual_components(theta)
    log_gamma, log_n = theta[:2]
    n_fit = np.exp(log_n)
    lam_fit = Vratio ** (1.0 / (3.0 * n_fit))
    C_fit = lam_fit**2
    P_model = traction / C_fit

    return {
        "time": time,
        "volume": volume,
        "pressure_data": Pout,
        "pressure_model": P_model,
        "pressure_residual": P_model - Pout,
        "fit_mask": fit_mask,
    }


def plot_fitted_model_against_data(plot_result):
    plt.figure(figsize=(7, 5))
    plt.plot(plot_result["volume"], plot_result["pressure_data"], "-", markersize=3, label="input PV data")
    plt.plot(plot_result["volume"], plot_result["pressure_model"], "-", linewidth=2, label="fitted spherical model")
    if FIT_BRANCH != "all":
        m = plot_result["fit_mask"]
        plt.plot(plot_result["volume"][m], plot_result["pressure_data"][m], "o", markersize=5, label=f"points used: {FIT_BRANCH}")
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (Barye)")
    #plt.title("PV fit: fitted model vs uploaded/input data")
    #plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("0DModeling/calibrated0DModel.png", dpi=300)
    plt.show()

    # plt.figure(figsize=(8, 5))
    # plt.plot(plot_result["time"], plot_result["pressure_data"], "o-", markersize=3, label="input pressure data")
    # plt.plot(plot_result["time"], plot_result["pressure_model"], "-", linewidth=2, label="fitted model pressure")
    # plt.xlabel("Time")
    # plt.ylabel("Pressure")
    # plt.title("Pressure fit over time")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(8, 4))
    # plt.axhline(0.0, linewidth=1)
    # plt.plot(plot_result["time"], plot_result["pressure_residual"], "o-", markersize=3)
    # plt.xlabel("Time")
    # plt.ylabel("P_model - P_data")
    # plt.title("Pressure residual over time")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()


if PLOT_FITTED_MODEL_VS_DATA:
    fit_plot = compute_fitted_model_for_plot(sol.x)
    plot_fitted_model_against_data(fit_plot)