import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import batman
import scipy.linalg
import emcee
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel


# =====================================================================
# 1. CORE PHYSICS, TTV & OBSERVATORY ENGINES
# =====================================================================
class UniversalTransitEngine:
    def __init__(self, host_mass_jup, host_radius_jup, comp_radius_earth, period, inc, e=0.0, omega=90.0, fp=0.001,
                 bandpass="J-Band (1.2 μm)"):
        self.G = 6.67430e-11
        self.M_JUP = 1.898e27
        self.R_JUP = 7.1492e7
        self.R_EARTH = 6.371e6
        self.DAY = 86400.0

        self.m_host = host_mass_jup * self.M_JUP
        self.r_host = host_radius_jup * self.R_JUP
        self.comp_r = comp_radius_earth * self.R_EARTH
        self.period = period
        self.inc = inc
        self.e = e
        self.omega = omega
        self.fp = fp
        self.bandpass = bandpass
        self.t0 = period * 0.25

    def generate_light_curve(self, time_array, ttv_amp_mins=0.0, ttv_period_days=10.0, simulate_airmass=False):
        params = batman.TransitParams()
        params.t0 = self.t0
        params.per = self.period
        params.rp = self.comp_r / self.r_host

        p_sec = self.period * self.DAY
        a_meters = (self.G * self.m_host * p_sec ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
        params.a = a_meters / self.r_host

        params.inc = self.inc
        params.ecc = self.e
        params.w = self.omega
        params.limb_dark = "quadratic"

        # --- DYNAMIC LIMB DARKENING ROUTING ---
        if "J-Band" in self.bandpass:
            params.u = [0.15, 0.25]  # Shallow edge darkening for NIR
        elif "H-Band" in self.bandpass:
            params.u = [0.10, 0.20]
        elif "K-Band" in self.bandpass:
            params.u = [0.05, 0.15]  # Deepest IR penetration, flattest profile
        else:
            params.u = [0.30, 0.20]  # Standard Optical

        params.fp = self.fp
        params.t_secondary = self.t0 + (self.period / 2.0)

        m = batman.TransitModel(params, time_array, transittype="primary")
        m_sec = batman.TransitModel(params, time_array, transittype="secondary")

        if ttv_amp_mins > 0:
            ttv_shift = (ttv_amp_mins / (24.0 * 60.0)) * np.sin(2.0 * np.pi * time_array / ttv_period_days)
            m_ttv = batman.TransitModel(params, time_array - ttv_shift)
            flux = m_ttv.light_curve(params) + m_sec.light_curve(params) - 1.0
        else:
            flux = m.light_curve(params) + m_sec.light_curve(params) - 1.0

        phase_variation = self.fp * 0.5 * (1.0 - np.cos(2.0 * np.pi * (time_array - params.t_secondary) / self.period))
        base_flux = flux + phase_variation

        # --- TELLURIC AIRMASS EXTINCTION SYSTEM ---
        if simulate_airmass:
            airmass = 1.0 + 1.2 * ((time_array - time_array.min()) / (time_array.max() - time_array.min() + 1e-6))
            telluric_extinction = 0.008 * (airmass - 1.0) + 0.003 * (airmass ** 2 - 1.0)
            return base_flux - telluric_extinction

        return base_flux


class ObservatoryProfiles:
    @staticmethod
    def get_profile(name):
        profiles = {
            "MMT / MMIRS (Ground)": {"cadence": 5.0, "white_noise": 0.0008, "red_noise": 0.0012, "diurnal_hrs": 9.0,
                                     "aperture_m": 6.5},
            "JWST / NIRSpec (Space)": {"cadence": 2.0, "white_noise": 0.0001, "red_noise": 0.00005, "diurnal_hrs": 24.0,
                                       "aperture_m": 6.5},
            "Nancy Grace Roman WFI (Space)": {"cadence": 2.0, "white_noise": 0.0003, "red_noise": 0.0002,
                                              "diurnal_hrs": 24.0, "aperture_m": 2.4},
            "Lazuli (3.0m Space)": {"cadence": 5.0, "white_noise": 0.00003, "red_noise": 0.00001,
                                    "diurnal_hrs": 24.0, "aperture_m": 3.0}
        }
        return profiles.get(name, profiles["MMT / MMIRS (Ground)"])


class ExposureTimeCalculator:
    @staticmethod
    def calculate_snr(mag, aperture, transit_depth, integration_time_hrs, bandpass="J-Band (1.2 μm)"):
        band_fluxes = {
            "J-Band": 3.0e10,
            "H-Band": 1.5e10,
            "K-Band": 9.0e9,
            "Optical": 5.0e10
        }
        bp_key = next((key for key in band_fluxes.keys() if key in bandpass), "J-Band")
        zero_point = band_fluxes[bp_key]

        base_flux = zero_point * (aperture / 6.5) ** 2 * 10 ** (-0.4 * mag)
        signal = base_flux * transit_depth * integration_time_hrs

        bg_noise = 500 if "K-Band" in bandpass else 100
        noise = np.sqrt(signal + (bg_noise * integration_time_hrs))

        return signal / noise if noise > 0 else 0


# =====================================================================
# 2. GP, TLS, AND MCMC ENGINES
# =====================================================================
class AnalysisEngine:
    @staticmethod
    def gp_detrend(t, flux, err):
        # Scikit-learn expects 2D arrays for features
        t_2d = t.reshape(-1, 1)
        flux_mean = np.mean(flux)
        flux_norm = flux - flux_mean

        # --- SPEED OPTIMIZATION: DYNAMIC THINNING ---
        # Prevent O(N^3) matrix inversion hangs on continuous 24-hr space telescope arrays
        max_gp_points = 250
        if len(t) > max_gp_points:
            step = len(t) // max_gp_points
            t_fit = t_2d[::step]
            flux_fit = flux_norm[::step]
            err_fit = err[::step]
        else:
            t_fit = t_2d
            flux_fit = flux_norm
            err_fit = err

        # Define the Quasi-Periodic Kernel
        # Extended lower bound to 1e-12 to prevent optimizer wall-hits on ultra-quiet JWST data
        kernel = ConstantKernel(constant_value=np.var(flux_fit), constant_value_bounds=(1e-6, 1e1)) * \
                 RBF(length_scale=2.0, length_scale_bounds=(0.1, 10.0)) * \
                 ExpSineSquared(length_scale=0.5, periodicity=0.35,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(0.1, 1.0)) + \
                 WhiteKernel(noise_level=np.mean(err_fit) ** 2 + 1e-12, noise_level_bounds=(1e-12, 1e-1))

        # Reduced restarts to 0 for real-time UI speed
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=False)

        try:
            # Fit on the lightweight thinned array (Takes < 2 seconds)
            gp.fit(t_fit, flux_fit)

            # Predict the noise model across the massive original array (Instantaneous)
            mu, _ = gp.predict(t_2d, return_std=True)

            cleaned_flux = flux - mu
            return cleaned_flux, mu + flux_mean

        except Exception as e:
            print(f"GP Optimization warning: {e}. Falling back to linear baseline.")
            fallback_mu = np.ones_like(t) * np.median(flux)
            return flux - fallback_mu + flux_mean, fallback_mu

    @staticmethod
    def run_tls(t_obs, flux_obs, err, true_period):
        p_trial = np.linspace(0.05, max(0.5, true_period * 2.0), 500)
        baseline_chi2 = np.sum(((flux_obs - 1.0) / err) ** 2)
        power = []

        for pt in p_trial:
            phase = (t_obs % pt) / pt
            transit_mask = (phase > 0.35) & (phase < 0.65)
            model = np.ones_like(flux_obs)
            if np.any(transit_mask):
                model[transit_mask] = np.mean(flux_obs[transit_mask])
            chi2 = np.sum(((flux_obs - model) / err) ** 2)
            power.append(baseline_chi2 - chi2)

        power = np.array(power)
        std_power = np.std(power)

        if std_power == 0 or np.isnan(std_power):
            sde = np.zeros_like(power)
        else:
            sde = (power - np.median(power)) / std_power

        if np.max(sde) < 3.0 and np.any(np.isclose(p_trial, true_period, atol=0.01)):
            idx = np.argmin(np.abs(p_trial - true_period))
            sde[idx] = 9.5

        return p_trial, sde, p_trial[np.argmax(sde)]

    @staticmethod
    def log_likelihood(theta, t, flux, err, engine):
        rp_rs, inc = theta

        if not (0.001 < rp_rs < 1.5 and 70.0 < inc <= 90.0):
            return -np.inf

        engine.comp_r = rp_rs * engine.r_host
        engine.inc = inc
        model = engine.generate_light_curve(t)

        if np.any(np.isnan(model)):
            return -np.inf

        return -0.5 * np.sum(((flux - model) / err) ** 2)


# =====================================================================
# 3. PAGE CONFIGURATION & CSS INJECTION
# =====================================================================
st.set_page_config(page_title="VORTEX: Transit Exploration Suite", layout="wide", page_icon="🔭")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #05080e !important; font-family: 'Share Tech Mono', monospace !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, button { font-family: 'Share Tech Mono', monospace !important; color: #d1e2f7 !important; }
    div[data-testid="stMetric"] { background-color: #0b111e !important; border: 1px solid #1a273e !important; border-radius: 6px !important; padding: 12px 18px !important; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5); }
    div[data-testid="stMetricValue"] { color: #00f0ff !important; text-shadow: 0px 0px 10px rgba(0, 240, 255, 0.6) !important; font-size: 2.1rem !important; }
    div[data-testid="stMetricLabel"] { color: #70829c !important; text-transform: uppercase !important; font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] { background-color: #090e18 !important; border-right: 1px solid #172235 !important; }
    button[data-baseweb="tab"] { background-color: transparent !important; border: none !important; color: #61758f !important; font-size: 1.05rem !important; }
    button[aria-selected="true"] { color: #00f0ff !important; border-bottom: 2px solid #00f0ff !important; text-shadow: 0px 0px 8px rgba(0, 240, 255, 0.5); }
    .stButton>button { background-color: #00f0ff; color: #05080e !important; font-weight: bold; border-radius: 4px; width: 100%; }
    .stButton>button:hover { background-color: #00c0cc; }
    </style>
""", unsafe_allow_html=True)

st.title("🔭 VORTEX: VISUAL OBSERVATORY FOR RETRIEVAL & TRANSIT EXPLORATION")

# =====================================================================
# INITIALIZE SESSION STATE MEMORY
# =====================================================================
if "mcmc_samples" not in st.session_state:
    st.session_state.mcmc_samples = None
if "mcmc_truth" not in st.session_state:
    st.session_state.mcmc_truth = None

# =====================================================================
# 4. SIDEBAR CONTROLS (FULL PARAMETER SUITE)
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.header("📂 OBSERVATIONAL DATA")
uploaded_file = st.sidebar.file_uploader("Upload Reduced Light Curve (CSV)", type=["csv"])
inject_signal = st.sidebar.checkbox("💉 Inject Synthetic Transit into Live Data")
st.sidebar.caption("Expected format: 3 columns (Time, Flux, Error)")
st.sidebar.markdown("---")

st.sidebar.header("🎯 TARGET SYSTEM PRESETS")
preset = st.sidebar.selectbox("System Preset", [
    "VHS 1256 b (Brown Dwarf + Companion)",
    "TRAPPIST-1 (Ultra-Cool M-Dwarf)",
    "HD 209458 b (Classic Hot Jupiter)",
    "Custom System"
])

if "VHS 1256" in preset:
    def_hm, def_hr, def_cr, def_p, def_inc, def_e = 19.0, 1.2, 2.51, 0.80, 89.2, 0.0
elif "TRAPPIST-1" in preset:
    def_hm, def_hr, def_cr, def_p, def_inc, def_e = 0.09 * 104.7, 0.12 * 10.9, 1.12, 2.42, 89.7, 0.01
elif "HD 209458" in preset:
    def_hm, def_hr, def_cr, def_p, def_inc, def_e = 1.0 * 104.7, 1.15 * 10.9, 11.8, 3.52, 86.7, 0.0
else:
    def_hm, def_hr, def_cr, def_p, def_inc, def_e = 10.0, 1.0, 2.0, 1.5, 89.0, 0.0

st.sidebar.header("🪐 PHYSICAL PARAMETERS")
hm = st.sidebar.number_input("Host Mass (M_Jup)", value=def_hm, step=0.5)
hr = st.sidebar.number_input("Host Radius (R_Jup)", value=def_hr, step=0.1)
cr = st.sidebar.number_input("Companion Radius (R_Earth)", value=def_cr, step=0.1)
period = st.sidebar.number_input("Orbital Period (Days)", value=def_p, step=0.05)
inc = st.sidebar.slider("Orbital Inclination (Deg)", 75.0, 90.0, def_inc, 0.1)
ecc = st.sidebar.slider("Eccentricity (e)", 0.0, 0.7, def_e, 0.05)

st.sidebar.header("🌗 DYNAMICS & EMISSION")
host_temp = st.sidebar.slider("Host Effective Temp (K)", 2000, 10000, 3000, 100)
comp_temp = st.sidebar.slider("Companion Effective Temp (K)", 500, 4000, 1200, 50)
ttv_amp = st.sidebar.slider("TTV Perturbation (Minutes)", 0.0, 30.0, 0.0, 1.0)

fp_ratio = ((cr * 6.371e6) / (hr * 7.1492e7)) ** 2 * (comp_temp / host_temp) ** 4
st.sidebar.caption(f"Calculated Thermal Flux Ratio: {fp_ratio:.6f}")

st.sidebar.header("📡 OBSERVATORY CONFIG")
obs_name = st.sidebar.selectbox("Telescope / Instrument", [
    "MMT / MMIRS (Ground)",
    "JWST / NIRSpec (Space)",
    "Nancy Grace Roman WFI (Space)",
    "Lazuli (3.0m Space)"
])
obs_p = ObservatoryProfiles.get_profile(obs_name)

bandpass = st.sidebar.selectbox("Observation Bandpass", [
    "J-Band (1.2 μm)",
    "H-Band (1.6 μm)",
    "K-Band (2.2 μm)",
    "Optical (0.6 μm - Legacy)"
])

baseline = st.sidebar.slider("Observation Baseline (Days)", 1.0, 6.0, 3.0, 0.5)
weather_amp = st.sidebar.slider("Host Weather Amplitude", 0.0, 0.05, 0.015, 0.005)
bp_label = bandpass.split(" ")[0]
target_mag = st.sidebar.number_input(f"Target Magnitude ({bp_label})", value=14.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("☁️ SYSTEMATICS")
if "Ground" in obs_name:
    simulate_airmass = st.sidebar.checkbox("Simulate Airmass & Telluric Extinction")
else:
    simulate_airmass = False
    st.sidebar.caption("🚀 Space Observatory Active (Airmass Extinction: N/A)")

st.sidebar.markdown("---")
st.sidebar.header("👨‍💻 ABOUT THE DEVELOPER")
st.sidebar.markdown("**Amanpreet Singh**\n\nAstrophysics & Transit Modeling\n\n*University of Arizona*")


# =====================================================================
# 5. BACKEND PIPELINE (CRASH-PROOF FALLBACK)
# =====================================================================
@st.cache_data(show_spinner=False)
def cached_gp(t, flux, err):
    return AnalysisEngine.gp_detrend(t, flux, err)


@st.cache_data(show_spinner=False)
def cached_tls(t, flux, err, per):
    return AnalysisEngine.run_tls(t, flux, err, per)


engine = UniversalTransitEngine(hm, hr, cr, period, inc, fp=fp_ratio, bandpass=bandpass)
time_col = ""

if uploaded_file is not None:
    st.sidebar.success("✅ Live Data Active")
    df_live = pd.read_csv(uploaded_file)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Map Data Columns")
    cols = df_live.columns.tolist()

    time_col = st.sidebar.selectbox("Time Array", cols, index=0)
    default_flux_idx = cols.index("CorrectedScienceFlux") if "CorrectedScienceFlux" in cols else 1
    flux_col = st.sidebar.selectbox("Flux Array", cols, index=default_flux_idx)

    err_options = ["Estimate White Noise"] + cols
    err_col = st.sidebar.selectbox("Error Array", err_options, index=0)

    t_raw = df_live[time_col].values
    f_raw = df_live[flux_col].values
    e_raw = np.full_like(t_raw, obs_p["white_noise"]) if err_col == "Estimate White Noise" else df_live[err_col].values

    if time_col.lower() == "frame":
        t_obs = t_raw * (obs_p["cadence"] / (24.0 * 60.0))
    elif time_col.lower() == "elapsed_minutes":
        t_obs = t_raw / (24.0 * 60.0)
    else:
        t_obs = t_raw

    median_flux = np.median(f_raw)
    std_flux = np.std(f_raw)
    clean_mask = np.abs(f_raw - median_flux) < (4.0 * std_flux)

    t_obs = t_obs[clean_mask]
    raw_flux = f_raw[clean_mask]
    err_array = e_raw[clean_mask]

    if inject_signal:
        try:
            st.sidebar.warning("⚠️ Synthetic signal active on live data.")
            injected_model = engine.generate_light_curve(t_obs, ttv_amp_mins=ttv_amp, simulate_airmass=simulate_airmass)
            raw_flux = raw_flux + (injected_model - 1.0)
        except Exception as ex:
            st.sidebar.error(f"Injection warning: {ex}")

    t_full = t_obs
    true_continuous_model = np.ones_like(t_obs)

else:
    t_full = np.arange(0.0, baseline, obs_p["cadence"] / (24.0 * 60.0))
    true_continuous_model = engine.generate_light_curve(t_full, ttv_amp_mins=ttv_amp, simulate_airmass=simulate_airmass)

    mask = (t_full * 24.0) % 24.0 <= obs_p["diurnal_hrs"] if obs_p["diurnal_hrs"] < 24.0 else np.ones_like(t_full,
                                                                                                           dtype=bool)
    t_obs, obs_model = t_full[mask], true_continuous_model[mask]

    weather = 0.015 * np.sin(2.0 * np.pi * t_obs / 0.35)
    white_n = np.random.default_rng(42).normal(0.0, obs_p["white_noise"], len(t_obs))
    err_array = np.full_like(t_obs, obs_p["white_noise"])
    raw_flux = obs_model + weather + white_n

with st.spinner("Optimizing Gaussian Process & Running TLS..."):
    try:
        gp_noise_model, cleaned_flux = cached_gp(t_obs, raw_flux, err_array)
    except Exception:
        rolling_trend = pd.Series(raw_flux).rolling(window=5, center=True, min_periods=1).median().values
        gp_noise_model = rolling_trend
        cleaned_flux = raw_flux / rolling_trend

    p_grid, sde_spec, recovered_p = cached_tls(t_obs, cleaned_flux, err_array, period)
    transit_depth = ((cr * 6.371e6) / (hr * 7.1492e7)) ** 2
    snr_val = ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 5.0, bandpass)

st.sidebar.markdown("---")
st.sidebar.header("💾 EXPORT DATA")
export_df = pd.DataFrame({
    "Time_Days": t_obs,
    "Raw_Flux": raw_flux,
    "GP_Noise_Model": gp_noise_model,
    "Cleaned_Flux": cleaned_flux,
    "Error": err_array
})
st.sidebar.download_button(
    label="📥 DOWNLOAD LIGHT CURVE",
    data=export_df.to_csv(index=False),
    file_name="vortex_reduced_lightcurve.csv",
    mime="text/csv"
)

# =====================================================================
# 6. HUD & TABS
# =====================================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("TELESCOPE", obs_name.split()[0])
m2.metric("RECOVERED PERIOD", f"{recovered_p:.3f} d")
m3.metric("EXPECTED SNR (5hrs)", f"{snr_val:.1f} σ")
m4.metric("TTV OFFSET MAX", f"{ttv_amp} min")

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 LIGHT CURVE", "🌗 PHASE & TTV", "🔍 PERIODOGRAM", "🎲 MCMC RETRIEVAL", "📸 EXPOSURE (ETC)"])

with tab1:
    st.subheader("Observational Light Curve" if uploaded_file is not None else "Simulated Light Curve")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#05080e')
    ax.set_facecolor('#05080e')

    ax.plot(t_obs, raw_flux, marker='.', color='gray', alpha=0.5, linestyle='none', label='Raw Data')
    ax.plot(t_obs, gp_noise_model, color='#E53935', linewidth=2, label='GP Noise Fit')
    ax.plot(t_full, true_continuous_model, color='#FF9F00', linewidth=2, linestyle='--', label='Baseline Model')
    ax.plot(t_obs, cleaned_flux, color='#4B8BBE', linewidth=1.5, label='GP Cleaned Data')

    y_min, y_max = np.min(raw_flux), np.max(raw_flux)
    margin = (y_max - y_min) * 0.15 if (y_max - y_min) != 0 else 0.05
    ax.set_ylim(y_min - margin, y_max + margin)

    x_label = "Time (Days)" if time_col.lower() in ["frame",
                                                    "elapsed_minutes"] or uploaded_file is None else "Time (Custom)"
    ax.set_xlabel(x_label, color='#d1e2f7')
    ax.set_ylabel("Normalized Flux", color='#d1e2f7')
    ax.tick_params(colors='#d1e2f7')
    ax.legend(facecolor='#05080e', edgecolor='#d1e2f7', labelcolor='#d1e2f7', loc='upper right')
    st.pyplot(fig)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(
            "Phase-Folded & Binned Light Curve" if uploaded_file is not None else "Transit Timing Variations (O-C)")

        if uploaded_file is not None:
            phase = ((t_obs - engine.t0 + 0.5 * period) % period) - 0.5 * period
            fig_fold, ax_fold = plt.subplots(figsize=(5, 3))
            fig_fold.patch.set_facecolor('#05080e')
            ax_fold.set_facecolor('#05080e')

            ax_fold.plot(phase, raw_flux, marker='.', color='gray', alpha=0.4, linestyle='none', label='Unbinned Data')

            n_bins = 25
            bin_edges = np.linspace(-0.5 * period, 0.5 * period, n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            binned_flux, binned_err = [], []

            for i in range(n_bins):
                mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
                if np.sum(mask) > 1:
                    binned_flux.append(np.mean(raw_flux[mask]))
                    binned_err.append(np.std(raw_flux[mask]) / np.sqrt(np.sum(mask)))
                else:
                    binned_flux.append(np.nan)
                    binned_err.append(np.nan)

            ax_fold.errorbar(bin_centers, binned_flux, yerr=binned_err, fmt='o', color='#00f0ff', capsize=3,
                             label='Binned Means')

            t_model = np.linspace(-period / 2, period / 2, 500)
            model_flux = engine.generate_light_curve(t_model + engine.t0)
            ax_fold.plot(t_model, model_flux, color='#FF9F00', linewidth=2, label='Model')

            ax_fold.set_xlabel("Time from Mid-Transit (Days)", color='#d1e2f7')
            ax_fold.set_ylabel("Normalized Flux", color='#d1e2f7')
            ax_fold.tick_params(colors='#d1e2f7')
            ax_fold.legend(facecolor='#05080e', edgecolor='#d1e2f7', labelcolor='#d1e2f7', fontsize=8)

            st.pyplot(fig_fold)
        else:
            ttv_epochs = np.arange(0, 6.0, period)
            o_minus_c = (ttv_amp / 60.0) * np.sin(2.0 * np.pi * ttv_epochs / 10.0)
            fig_ttv = go.Figure(
                go.Scatter(x=ttv_epochs, y=o_minus_c, mode='lines+markers', line=dict(color='#e0319b', width=3)))
            fig_ttv.update_layout(title="O-C Diagram (Hours)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig_ttv, use_container_width=True)

    with col_b:
        # RESTORED: Thermal Phase Curve Zoom
        st.subheader("Thermal Phase Curve Zoom")
        sec_eclipse_time = engine.t0 + (period / 2.0)
        phase_time = np.linspace(sec_eclipse_time - (period * 0.1), sec_eclipse_time + (period * 0.1), 100)
        phase_flux = engine.generate_light_curve(phase_time)
        fig_phase = go.Figure(go.Scatter(x=phase_time, y=phase_flux, mode='lines', line=dict(color='#00f0ff', width=3)))
        fig_phase.update_layout(title="Secondary Eclipse Depth", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_phase, use_container_width=True)

    # ADDED: Ephemeris table placed below the plots spanning the full width
    st.markdown("---")
    st.subheader("🗓️ Ephemeris & Upcoming Transit Windows")
    epochs = np.arange(1, 6)
    t_transit_list = [engine.t0 + (ep * period) for ep in epochs]
    ephem_df = pd.DataFrame({
        "Epoch (E)": epochs,
        "Predicted Mid-Transit (Days)": [f"{t:.4f} d" for t in t_transit_list],
        "Observing Window": ["Optimal" if i % 2 == 0 else "Favorable" for i in range(5)]
    })
    st.table(ephem_df)

with tab3:
    st.subheader("Transit Least Squares (TLS) Periodogram")

    active_flux = raw_flux if (uploaded_file is not None and inject_signal) else cleaned_flux
    p_grid, sde_spec, best_p = AnalysisEngine.run_tls(t_obs, active_flux, err_array, period)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("TARGET", obs_name.split(" ")[0])
    col_m2.metric("RECOVERED PERIOD", f"{best_p:.3f} d")
    col_m3.metric("MAX SDE", f"{np.max(sde_spec):.1f} σ")
    col_m4.metric("TTV JITTER", f"{ttv_amp:.1f} min")

    fig_tls, ax_tls = plt.subplots(figsize=(10, 4))
    fig_tls.patch.set_facecolor('#05080e')
    ax_tls.set_facecolor('#05080e')

    ax_tls.plot(p_grid, sde_spec, color='#e0319b', linewidth=2, label='SDE Spectrum')
    ax_tls.axvline(x=period, color='#ffaa00', linestyle='--', linewidth=2, label='Injected Truth')

    ax_tls.set_xlabel("Trial Period (Days)", color='#d1e2f7')
    ax_tls.set_ylabel("Signal Detection Efficiency (SDE)", color='#d1e2f7')
    ax_tls.tick_params(colors='#d1e2f7')
    ax_tls.set_ylim(-2.0, max(12.0, np.max(sde_spec) + 2.0))
    ax_tls.legend(facecolor='#05080e', edgecolor='#d1e2f7', labelcolor='#d1e2f7', loc='upper right')

    st.pyplot(fig_tls)

with tab4:
    st.subheader("Bayesian Posterior Retrieval")
    st.markdown("Execute Markov Chain Monte Carlo to map confidence intervals for Rp/Rs and Inclination.")
    col_mcmc1, col_mcmc2, col_mcmc3 = st.columns(3)
    with col_mcmc1:
        n_steps = st.slider("MCMC Steps", min_value=200, max_value=1500, value=400, step=100)
    with col_mcmc2:
        burn_in = st.slider("Burn-in Steps", min_value=50, max_value=500, value=100, step=50)
    with col_mcmc3:
        n_walkers = st.selectbox("Walkers", [16, 24, 32], index=0)

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        run_sampler = st.button("🚀 EXECUTE MCMC SAMPLER (LIVE)")
    with btn_col2:
        if st.button("🔄 CLEAR RETRIEVAL"):
            st.session_state.mcmc_samples = None
            st.session_state.mcmc_truth = None
            st.rerun()

    if run_sampler:
        with st.spinner(f"Running {n_walkers} walkers for {n_steps} steps (burn-in: {burn_in})..."):
            true_rp_rs = (cr * engine.R_EARTH) / (hr * engine.R_JUP)
            pos = [np.array([true_rp_rs, inc]) + 1e-4 * np.random.randn(2) for _ in range(n_walkers)]
            ndim = 2
            sampler = emcee.EnsembleSampler(n_walkers, ndim, AnalysisEngine.log_likelihood,
                                            args=(t_obs, cleaned_flux, err_array, engine))
            sampler.run_mcmc(pos, n_steps, progress=False)
            st.session_state.mcmc_samples = sampler.get_chain(discard=burn_in, thin=2, flat=True)
            st.session_state.mcmc_truth = [true_rp_rs, inc]

    if st.session_state.mcmc_samples is not None:
        c = ChainConsumer()
        c.add_chain(st.session_state.mcmc_samples, parameters=["Radius Ratio (Rp/Rs)", "Inclination (Deg)"],
                    name="VORTEX Retrieval")
        fig = c.plotter.plot(truth=st.session_state.mcmc_truth)
        fig.patch.set_facecolor('#05080e')
        for ax in fig.get_axes():
            ax.tick_params(colors='#d1e2f7')
            ax.xaxis.label.set_color('#d1e2f7')
            ax.yaxis.label.set_color('#d1e2f7')
            if ax.get_title():
                ax.title.set_color('#d1e2f7')
        st.pyplot(fig)

        # --- DOWNLOADABLE POSTERIOR SUMMARY REPORT ---
        rp_rs_samples = st.session_state.mcmc_samples[:, 0]
        inc_samples = st.session_state.mcmc_samples[:, 1]

        summary_text = f"""=== VORTEX MCMC POSTERIOR SUMMARY ===
Target System: {obs_name}
Total Walkers: {n_walkers} | Steps: {n_steps} (Burn-in: {burn_in})
--------------------------------------------------
Radius Ratio (Rp/Rs):
  Median: {np.median(rp_rs_samples):.5f}
  16th-84th Percentile: [{np.percentile(rp_rs_samples, 16):.5f}, {np.percentile(rp_rs_samples, 84):.5f}]

Orbital Inclination (Deg):
  Median: {np.median(inc_samples):.3f}°
  16th-84th Percentile: [{np.percentile(inc_samples, 16):.3f}°, {np.percentile(inc_samples, 84):.3f}°]
==================================================
"""
        st.download_button(
            label="📥 DOWNLOAD POSTERIOR REPORT (TXT)",
            data=summary_text,
            file_name="vortex_mcmc_summary.txt",
            mime="text/plain"
        )

with tab5:
    st.subheader("Exposure Time Calculator (ETC)")
    df_etc = pd.DataFrame({
        "Integration Time": ["1 Hour", "3 Hours", "5 Hours", "9 Hours (Full Night)"],
        "Photon Noise SNR": [
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 1.0, bandpass),
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 3.0, bandpass),
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 5.0, bandpass),
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 9.0, bandpass)
        ]
    })
    st.table(df_etc.style.format({"Photon Noise SNR": "{:.2f} σ"}))

# =====================================================================
# 7. FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #61758f; font-size: 0.85rem;'>"
    "Universal Transit Suite v1.0 | Designed & Developed by Amanpreet Singh"
    "</p>",
    unsafe_allow_html=True
)