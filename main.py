import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import batman
import scipy.linalg
import emcee
import corner
import matplotlib.pyplot as plt


# =====================================================================
# 1. CORE PHYSICS, TTV & OBSERVATORY ENGINES
# =====================================================================
class UniversalTransitEngine:
    def __init__(self, host_mass_jup, host_radius_jup, comp_radius_earth, period, inc, e=0.0, omega=90.0, fp=0.001):
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
        self.fp = fp  # Planet-to-star flux ratio for secondary eclipse

        # Absolute anchor for transit time to prevent phase curve sliding
        self.t0 = period * 0.25

    def generate_light_curve(self, time_array, ttv_amp_mins=0.0, ttv_period_days=10.0):
        params = batman.TransitParams()
        params.t0 = self.t0  # Using anchored time
        params.per = self.period
        params.rp = self.comp_r / self.r_host

        p_sec = self.period * self.DAY
        a_meters = (self.G * self.m_host * p_sec ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
        params.a = a_meters / self.r_host

        params.inc = self.inc
        params.ecc = self.e
        params.w = self.omega
        params.limb_dark = "quadratic"
        params.u = [0.3, 0.2]

        # Secondary Eclipse / Phase Curve parameters
        params.fp = self.fp
        params.t_secondary = self.t0 + (self.period / 2.0)

        # Generate base model
        m = batman.TransitModel(params, time_array, transittype="primary")
        m_sec = batman.TransitModel(params, time_array, transittype="secondary")

        # TTV Injection: Warp the time array slightly based on a perturbing body
        if ttv_amp_mins > 0:
            ttv_shift = (ttv_amp_mins / (24.0 * 60.0)) * np.sin(2.0 * np.pi * time_array / ttv_period_days)
            m_ttv = batman.TransitModel(params, time_array - ttv_shift)
            flux = m_ttv.light_curve(params) + m_sec.light_curve(params) - 1.0
        else:
            flux = m.light_curve(params) + m_sec.light_curve(params) - 1.0

        # Add thermal phase variation (sine wave matching orbital period)
        phase_variation = self.fp * 0.5 * (1.0 - np.cos(2.0 * np.pi * (time_array - params.t_secondary) / self.period))

        return flux + phase_variation


class ObservatoryProfiles:
    @staticmethod
    def get_profile(name):
        profiles = {
            "MMT / MMIRS (Ground)": {"cadence": 5.0, "white_noise": 0.0008, "red_noise": 0.0012, "diurnal_hrs": 9.0,
                                     "aperture_m": 6.5},
            "JWST / NIRSpec (Space)": {"cadence": 2.0, "white_noise": 0.0001, "red_noise": 0.00005, "diurnal_hrs": 24.0,
                                       "aperture_m": 6.5},
            "Nancy Grace Roman WFI (Space)": {"cadence": 2.0, "white_noise": 0.0003, "red_noise": 0.0002,
                                              "diurnal_hrs": 24.0, "aperture_m": 2.4}
        }
        return profiles.get(name, profiles["MMT / MMIRS (Ground)"])


class ExposureTimeCalculator:
    @staticmethod
    def calculate_snr(mag, aperture, transit_depth, integration_time_hrs):
        # Simplified photometric SNR estimator
        base_flux = 1e10 * (aperture / 6.5) ** 2 * 10 ** (-0.4 * mag)
        signal = base_flux * transit_depth * integration_time_hrs
        noise = np.sqrt(signal + (100 * integration_time_hrs))  # Shot noise + background
        return signal / noise if noise > 0 else 0


# =====================================================================
# 2. GP, TLS, AND MCMC ENGINES
# =====================================================================
class AnalysisEngine:
    @staticmethod
    def gp_detrend(t_obs, flux_obs, err, length_scale=0.15, amp=0.02):
        dist_sq = (t_obs[:, None] - t_obs[None, :]) ** 2
        K = (amp ** 2) * np.exp(-0.5 * dist_sq / (length_scale ** 2)) + np.diag(err ** 2 + 1e-6)
        try:
            L = scipy.linalg.cholesky(K, lower=True)
            alpha = scipy.linalg.solve(L.T, scipy.linalg.solve(L, flux_obs - 1.0))
            gp_mean = 1.0 + K @ alpha
        except scipy.linalg.LinAlgError:
            gp_mean = np.ones_like(t_obs)
        return flux_obs / gp_mean, gp_mean

    @staticmethod
    def run_tls(t_obs, flux_obs, err, true_period):
        p_trial = np.linspace(max(0.2, true_period - 0.5), true_period + 0.5, 300)
        baseline_chi2 = np.sum(((flux_obs - 1.0) / err) ** 2)
        power = []
        for pt in p_trial:
            phase = (t_obs % pt) / pt
            transit_mask = (phase > 0.45) & (phase < 0.55)
            model = np.ones_like(flux_obs)
            if np.any(transit_mask):
                model[transit_mask] = np.mean(flux_obs[transit_mask])
            power.append(baseline_chi2 - np.sum(((flux_obs - model) / err) ** 2))
        sde = (np.array(power) - np.median(power)) / (np.std(power) + 1e-12)
        return p_trial, sde, p_trial[np.argmax(sde)]

    @staticmethod
    def log_likelihood(theta, t, flux, err, engine):
        rp_rs, inc = theta

        # 1. Expand the prior bounds to safely accommodate massive Hot Jupiters
        if not (0.001 < rp_rs < 1.5 and 70.0 < inc <= 90.0):
            return -np.inf

        # 2. BUG FIX: Properly convert the unitless ratio back to absolute meters
        engine.comp_r = rp_rs * engine.r_host
        engine.inc = inc

        model = engine.generate_light_curve(t)

        # 3. Safety catch: If orbital geometry is physically impossible (e.g., planet inside star)
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
    h1, h2, h3, h4, h5, h6, p, span, div, label, button { font-family: 'Share Tech Mono', monospace !important; color: #d1e2f7 !important; }
    div[data-testid="stMetric"] { background-color: #0b111e !important; border: 1px solid #1a273e !important; border-radius: 6px !important; padding: 12px 18px !important; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5); }
    div[data-testid="stMetricValue"] { color: #00f0ff !important; text-shadow: 0px 0px 10px rgba(0, 240, 255, 0.6) !important; font-size: 2.1rem !important; }
    div[data-testid="stMetricLabel"] { color: #70829c !important; text-transform: uppercase !important; font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] { background-color: #090e18 !important; border-right: 1px solid #172235 !important; }
    button[data-baseweb="tab"] { background-color: transparent !important; border: none !important; color: #61758f !important; font-size: 1.05rem !important; }
    button[aria-selected="true"] { color: #00f0ff !important; border-bottom: 2px solid #00f0ff !important; text-shadow: 0px 0px 8px rgba(0, 240, 255, 0.5); }
    /* Style MCMC button */
    .stButton>button { background-color: #00f0ff; color: #05080e !important; font-weight: bold; border-radius: 4px; width: 100%; }
    .stButton>button:hover { background-color: #00c0cc; }
    </style>
""", unsafe_allow_html=True)

st.title("🔭 VORTEX: VISUAL OBSERVATORY FOR RETRIEVAL & TRANSIT EXPLORATION")

# =====================================================================
# 4. SIDEBAR CONTROLS (FULL PARAMETER SUITE)
# =====================================================================
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
ttv_amp = st.sidebar.slider("TTV Perturbation (Minutes)", 0.0, 30.0, 0.0, 1.0)
fp_ratio = st.sidebar.number_input("Flux Ratio fp (Secondary Eclipse)", value=0.0010, step=0.0005, format="%.4f")

st.sidebar.header("📡 OBSERVATORY CONFIG")
obs_name = st.sidebar.selectbox("Telescope / Instrument", [
    "MMT / MMIRS (Ground)",
    "JWST / NIRSpec (Space)",
    "Nancy Grace Roman WFI (Space)"
])
obs_p = ObservatoryProfiles.get_profile(obs_name)
baseline = st.sidebar.slider("Observation Baseline (Days)", 1.0, 15.0, 4.0, 0.5)
weather_amp = st.sidebar.slider("Host Weather Amplitude", 0.0, 0.05, 0.015, 0.005)
target_mag = st.sidebar.number_input("Target Magnitude (J-Band)", value=14.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("👨‍💻 ABOUT THE DEVELOPER")
st.sidebar.markdown(
    "**Amanpreet Singh**\n\n"
    "Astrophysics & Transit Modeling\n\n"
    "*University of Arizona*"
)

# =====================================================================
# 5. BACKEND PIPELINE
# =====================================================================
engine = UniversalTransitEngine(hm, hr, cr, period, inc, fp=fp_ratio)
t_full = np.arange(0.0, 6.0, obs_p["cadence"] / (24.0 * 60.0))
true_continuous_model = engine.generate_light_curve(t_full, ttv_amp_mins=ttv_amp)

mask = (t_full * 24.0) % 24.0 <= obs_p["diurnal_hrs"] if obs_p["diurnal_hrs"] < 24.0 else np.ones_like(t_full,
                                                                                                       dtype=bool)
t_obs, obs_model = t_full[mask], true_continuous_model[mask]

weather = 0.015 * np.sin(2.0 * np.pi * t_obs / 0.35)
white_n = np.random.default_rng(42).normal(0.0, obs_p["white_noise"], len(t_obs))
err_array = np.full_like(t_obs, obs_p["white_noise"])
raw_flux = obs_model + weather + white_n

cleaned_flux, _ = AnalysisEngine.gp_detrend(t_obs, raw_flux, err_array)
p_grid, sde_spec, recovered_p = AnalysisEngine.run_tls(t_obs, cleaned_flux, err_array, period)
transit_depth = ((cr * 6.371e6) / (hr * 7.1492e7)) ** 2
snr_val = ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, integration_time_hrs=5.0)

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
    fig_lc = go.Figure()
    fig_lc.add_trace(
        go.Scatter(x=t_obs, y=cleaned_flux, mode='markers', marker=dict(size=3.5, color='#00f0ff', opacity=0.6),
                   name="GP Cleaned Data"))
    fig_lc.add_trace(go.Scatter(x=t_full, y=true_continuous_model, mode='lines', line=dict(color='#ffaa00', width=2.5),
                                name="Model (Inc. Secondary)"))
    fig_lc.update_layout(xaxis=dict(title="Time (Days)", gridcolor="#172235"),
                         yaxis=dict(title="Normalized Flux", gridcolor="#172235"), template="plotly_dark",
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig_lc, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Transit Timing Variations (O-C)")
        ttv_epochs = np.arange(0, 6.0, period)
        o_minus_c = (ttv_amp / 60.0) * np.sin(2.0 * np.pi * ttv_epochs / 10.0)
        fig_ttv = go.Figure(
            go.Scatter(x=ttv_epochs, y=o_minus_c, mode='lines+markers', line=dict(color='#e0319b', width=3)))
        fig_ttv.update_layout(title="O-C Diagram (Hours)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_ttv, use_container_width=True)
    with col_b:
        st.subheader("Thermal Phase Curve Zoom")
        # Center the window dynamically based on the anchored t0
        sec_eclipse_time = engine.t0 + (period / 2.0)
        phase_time = np.linspace(sec_eclipse_time - (period * 0.1), sec_eclipse_time + (period * 0.1), 100)
        phase_flux = engine.generate_light_curve(phase_time)

        fig_phase = go.Figure(go.Scatter(x=phase_time, y=phase_flux, mode='lines', line=dict(color='#00f0ff', width=3)))
        fig_phase.update_layout(title="Secondary Eclipse Depth", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_phase, use_container_width=True)

with tab3:
    fig_tls = go.Figure(go.Scatter(x=p_grid, y=sde_spec, mode='lines', line=dict(color='#e0319b', width=2)))
    fig_tls.add_vline(x=period, line_dash="dash", line_color="#ffaa00", annotation_text="Injected Truth")
    fig_tls.update_layout(xaxis_title="Trial Period (Days)", yaxis_title="SDE", template="plotly_dark",
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig_tls, use_container_width=True)

with tab4:
    st.subheader("Bayesian Posterior Retrieval")
    st.markdown("Execute Markov Chain Monte Carlo to map confidence intervals for Rp/Rs and Inclination.")
    if st.button("🚀 EXECUTE MCMC SAMPLER (LIVE)"):
        with st.spinner("Initializing 16 Walkers... Burning in..."):
            true_rp_rs = (cr * engine.R_EARTH) / (hr * engine.R_JUP)
            pos = [np.array([true_rp_rs, inc]) + 1e-4 * np.random.randn(2) for i in range(16)]
            nwalkers, ndim = 16, 2
            sampler = emcee.EnsembleSampler(nwalkers, ndim, AnalysisEngine.log_likelihood,
                                            args=(t_obs, cleaned_flux, err_array, engine))
            sampler.run_mcmc(pos, 300, progress=False)  # Short chain for UI responsiveness

            flat_samples = sampler.get_chain(discard=50, thin=2, flat=True)

            fig = corner.corner(flat_samples, labels=["Radius Ratio (Rp/Rs)", "Inclination (Deg)"],
                                truths=[true_rp_rs, inc], color="#00f0ff", truth_color="#ffaa00")
            fig.patch.set_facecolor('#05080e')
            for ax in fig.get_axes():
                ax.tick_params(colors='#d1e2f7')
                ax.xaxis.label.set_color('#d1e2f7')
                ax.yaxis.label.set_color('#d1e2f7')
            st.pyplot(fig)

with tab5:
    st.subheader("Exposure Time Calculator (ETC)")
    df_etc = pd.DataFrame({
        "Integration Time": ["1 Hour", "3 Hours", "5 Hours", "9 Hours (Full Night)"],
        "Photon Noise SNR": [
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 1.0),
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 3.0),
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 5.0),
            ExposureTimeCalculator.calculate_snr(target_mag, obs_p["aperture_m"], transit_depth, 9.0)
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