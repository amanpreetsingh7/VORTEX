"""
VORTEX v2.0.2
Visual Observatory for Retrieval & Transit Exploration

Research-prototype transit simulation, injection/recovery, TLS search,
transit-aware Gaussian-process detrending, Bayesian retrieval, approximate
thermal phase modeling, and approximate exposure-time calculations.

Important scientific scope notes
--------------------------------
1. The ETC is intentionally labeled approximate. It is not a replacement for
   instrument-specific ETCs such as Pandeia/PandExo.
2. The thermal emission model uses blackbodies integrated over top-hat
   approximations to broad bands. Real brown-dwarf/planet spectra can differ
   strongly from blackbodies.
3. BJD_TDB conversion from UTC requires target coordinates and observatory
   coordinates.
4. Blind-search mode does not mask a known transit before the first TLS pass.
   It uses a robust preliminary detrend, then TLS, then transit-aware GP, then
   a second TLS pass.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import batman
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from transitleastsquares import transitleastsquares



# =============================================================================
# 0. CONSTANTS, BANDPASSES, HELPERS
# =============================================================================

MSUN_TO_MJUP = (const.M_sun / const.M_jup).decompose().value
RSUN_TO_RJUP = (const.R_sun / const.R_jup).decompose().value
REARTH_TO_RJUP = (const.R_earth / const.R_jup).decompose().value
DAY_S = 86400.0

BANDS = {
    "J-Band (1.2 μm)": {"center_um": 1.25, "width_um": 0.30, "vega_to_ab": 0.91},
    "H-Band (1.6 μm)": {"center_um": 1.65, "width_um": 0.30, "vega_to_ab": 1.39},
    "K-Band (2.2 μm)": {"center_um": 2.20, "width_um": 0.40, "vega_to_ab": 1.85},
    "Optical (0.6 μm)": {"center_um": 0.60, "width_um": 0.20, "vega_to_ab": 0.00},
}

APPROX_LD = {
    "J-Band (1.2 μm)": (0.15, 0.25),
    "H-Band (1.6 μm)": (0.10, 0.20),
    "K-Band (2.2 μm)": (0.05, 0.15),
    "Optical (0.6 μm)": (0.30, 0.20),
}


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy-version-safe trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based robust standard deviation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def safe_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return "unknown"


def quadratic_ld_is_physical(u1: float, u2: float) -> bool:
    # Common sufficient conditions for a physically sensible quadratic profile.
    return (u1 > 0.0) and (u1 + u2 < 1.0) and (u1 + 2.0 * u2 > 0.0)


def guess_time_format_from_column(column_name: str, values: np.ndarray) -> str:
    """
    Conservative name-based default for uploaded time columns.

    VORTEX deliberately does not infer an absolute Julian-date system merely
    from large numeric values. The selected column name controls the default,
    and the astronomer can override it under Advanced time handling.
    """
    name = str(column_name).strip().lower().replace("-", "_")
    compact = name.replace(" ", "")

    if "frame" in compact or compact in {"index", "framenumber", "frame_number"}:
        return "Frame Number"
    if "elapsed" in compact and ("minute" in compact or "min" in compact):
        return "Elapsed Minutes"
    if "elapsed" in compact and ("hour" in compact or "hr" in compact):
        return "Elapsed Hours"
    if "elapsed" in compact and "day" in compact:
        return "Elapsed Days"
    if "bjd" in compact:
        return "BJD_TDB"
    if "mjd" in compact:
        return "MJD_UTC"
    if compact in {"jd", "jdutc", "jd_utc", "juliandate", "julian_date"} or "jd_utc" in name:
        return "JD_UTC"

    # Unknown columns stay relative by default. This avoids silently treating
    # frame counters or arbitrary indices as absolute astronomical dates.
    return "Elapsed Days"


def default_flux_column_index(columns: list[str]) -> int:
    """Prefer reduced/normalized science flux rather than raw detector counts."""
    priorities = [
        "correctedscienceflux", "corrected_flux", "normalizedscience",
        "normalized_flux", "relative_flux", "flux",
    ]
    lowered = [str(c).replace(" ", "").lower() for c in columns]
    for target in priorities:
        target_compact = target.replace("_", "")
        for i, name in enumerate(lowered):
            if target_compact == name.replace("_", ""):
                return i
    for i, name in enumerate(lowered):
        if "correct" in name and "flux" in name:
            return i
    for i, name in enumerate(lowered):
        if "norm" in name and "flux" in name:
            return i
    return min(1, len(columns) - 1)


def boolean_bad_row_mask(series: pd.Series) -> np.ndarray:
    """Interpret common quality/outlier flags where True/non-zero means bad."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(np.mean(np.isfinite(numeric.to_numpy(dtype=float)))) if len(series) else 0.0
    if numeric_fraction > 0.8:
        values = numeric.to_numpy(dtype=float)
        return np.isfinite(values) & (values != 0.0)

    text = series.astype(str).str.strip().str.lower()
    return text.isin({
        "true", "1", "yes", "y", "bad", "outlier", "reject", "rejected",
        "remove", "removed", "invalid", "flagged",
    }).to_numpy(dtype=bool)


def phase_from_ephemeris(t: np.ndarray, t0: float, period: float) -> np.ndarray:
    return ((t - t0 + 0.5 * period) % period) - 0.5 * period


def parse_skycoord(ra_text: str, dec_text: str) -> SkyCoord:
    """
    Accept decimal-degree RA/Dec or sexagesimal RA + degree Dec.
    Examples:
        RA=330.795, Dec=18.884
        RA=22:03:10.81, Dec=+18:53:03.3
    """
    ra_text = str(ra_text).strip()
    dec_text = str(dec_text).strip()
    if ":" in ra_text or "h" in ra_text.lower():
        return SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg), frame="icrs")
    return SkyCoord(float(ra_text) * u.deg, float(dec_text) * u.deg, frame="icrs")


def to_internal_time(
    raw_time: np.ndarray,
    time_format: str,
    frame_cadence_min: float = 5.0,
    convert_utc_to_bjd: bool = False,
    coord: Optional[SkyCoord] = None,
    location: Optional[EarthLocation] = None,
) -> Tuple[np.ndarray, float, str, bool]:
    """
    Convert uploaded times to days and subtract a large absolute offset for
    numerical stability. Returns:
        internal_time_days, display_offset, display_label, is_absolute
    """
    raw = np.asarray(raw_time)
    values = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The selected time column contains no finite numeric values.")

    if time_format == "Frame Number":
        if not np.isfinite(frame_cadence_min) or frame_cadence_min <= 0:
            raise ValueError("Frame cadence must be a positive number of minutes.")
        values = (values - float(np.nanmin(finite))) * frame_cadence_min / 1440.0
        return values, 0.0, "Elapsed Days", False

    if time_format == "Elapsed Minutes":
        values = (values - np.nanmin(values)) / 1440.0
        return values, 0.0, "Elapsed Days", False

    if time_format == "Elapsed Hours":
        values = (values - np.nanmin(values)) / 24.0
        return values, 0.0, "Elapsed Days", False

    if time_format == "Elapsed Days":
        values = values - np.nanmin(values)
        return values, 0.0, "Elapsed Days", False

    if time_format == "BJD_TDB":
        finite = values[np.isfinite(values)]
        if finite.size and float(np.nanmedian(finite)) < 1_000_000:
            raise ValueError(
                "Selected BJD_TDB, but the time values are far smaller than a normal Julian Date. "
                "If this column contains frame numbers or relative times, choose Frame Number / Elapsed Days instead."
            )
        offset = float(np.floor(np.nanmin(values)))
        return values - offset, offset, "BJD_TDB", True

    if time_format in {"JD_UTC", "MJD_UTC"}:
        finite = values[np.isfinite(values)]
        if finite.size:
            med = float(np.nanmedian(finite))
            if time_format == "JD_UTC" and med < 1_000_000:
                raise ValueError("Selected JD_UTC, but these values do not look like full Julian Dates. Choose the correct relative/frame time format.")
            if time_format == "MJD_UTC" and med < 10_000:
                raise ValueError("Selected MJD_UTC, but these values do not look like Modified Julian Dates. Choose the correct relative/frame time format.")
        if convert_utc_to_bjd:
            if coord is None or location is None:
                raise ValueError("Target coordinates and observatory location are required for BJD_TDB conversion.")
            fmt = "jd" if time_format == "JD_UTC" else "mjd"
            times = Time(values, format=fmt, scale="utc", location=location)
            ltt = times.light_travel_time(coord, kind="barycentric")
            bjd = (times.tdb + ltt).jd
            offset = float(np.floor(np.nanmin(bjd)))
            return np.asarray(bjd - offset, dtype=float), offset, "BJD_TDB", True

        offset = float(np.floor(np.nanmin(values)))
        return values - offset, offset, time_format, True

    raise ValueError(f"Unsupported time format: {time_format}")


def display_time(internal_t: float, offset: float) -> float:
    return float(internal_t + offset)


# =============================================================================
# 1. BLACKBODY / TOP-HAT THERMAL MODEL
# =============================================================================

def planck_lambda(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    """Planck spectral radiance B_lambda; sufficient for flux-ratio work."""
    wavelength_m = np.asarray(wavelength_m, dtype=float)
    T = max(float(temperature_k), 1.0)
    h = const.h.value
    c = const.c.value
    k = const.k_B.value
    x = np.clip((h * c) / (wavelength_m * k * T), 1e-12, 700.0)
    return (2.0 * h * c**2 / wavelength_m**5) / np.expm1(x)


def calc_bandpass_flux_ratio(
    companion_temp_k: float,
    host_temp_k: float,
    companion_radius_m: float,
    host_radius_m: float,
    bandpass: str,
) -> float:
    """
    Approximate companion/host thermal flux ratio:
    blackbodies integrated through a top-hat bandpass.
    """
    cfg = BANDS[bandpass]
    center = cfg["center_um"] * 1e-6
    width = cfg["width_um"] * 1e-6
    lam_min = max(center - width / 2.0, 1e-9)
    lam_max = center + width / 2.0
    wav = np.linspace(lam_min, lam_max, 400)

    f_comp = _trapz(planck_lambda(wav, companion_temp_k), wav)
    f_host = _trapz(planck_lambda(wav, host_temp_k), wav)

    if not np.isfinite(f_comp) or not np.isfinite(f_host) or f_host <= 0:
        return 0.0

    return float((companion_radius_m / host_radius_m) ** 2 * (f_comp / f_host))


# =============================================================================
# 2. TRANSIT ENGINE
# =============================================================================

@dataclass(frozen=True)
class TransitEngine:
    host_mass_jup: float
    host_radius_jup: float
    companion_radius_earth: float
    period_days: float
    inclination_deg: float
    eccentricity: float = 0.0
    omega_deg: float = 90.0
    host_temp_k: float = 3000.0
    companion_temp_k: float = 1200.0
    bandpass: str = "J-Band (1.2 μm)"
    u1: float = 0.15
    u2: float = 0.25
    exposure_time_days: float = 0.0
    supersample_factor: int = 1
    nightside_fraction: float = 0.0

    @property
    def host_mass_kg(self) -> float:
        return float(self.host_mass_jup * const.M_jup.value)

    @property
    def host_radius_m(self) -> float:
        return float(self.host_radius_jup * const.R_jup.value)

    @property
    def companion_radius_m(self) -> float:
        return float(self.companion_radius_earth * const.R_earth.value)

    @property
    def host_mass_solar(self) -> float:
        return float(self.host_mass_jup / MSUN_TO_MJUP)

    @property
    def host_radius_solar(self) -> float:
        return float(self.host_radius_jup / RSUN_TO_RJUP)

    @property
    def rp_rs(self) -> float:
        return float(self.companion_radius_m / self.host_radius_m)

    @property
    def thermal_fp(self) -> float:
        return calc_bandpass_flux_ratio(
            self.companion_temp_k,
            self.host_temp_k,
            self.companion_radius_m,
            self.host_radius_m,
            self.bandpass,
        )

    def a_scaled(self, period_days: Optional[float] = None) -> float:
        P = float(self.period_days if period_days is None else period_days) * DAY_S
        a_m = (const.G.value * self.host_mass_kg * P**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
        return float(a_m / self.host_radius_m)

    def impact_parameter(
        self,
        inclination_deg: Optional[float] = None,
        period_days: Optional[float] = None,
    ) -> float:
        inc = math.radians(self.inclination_deg if inclination_deg is None else inclination_deg)
        a_rs = self.a_scaled(period_days)
        e = self.eccentricity
        w = math.radians(self.omega_deg)
        factor = (1.0 - e**2) / (1.0 + e * math.sin(w))
        return float(a_rs * math.cos(inc) * factor)

    def inclination_from_b(self, b: float, period_days: float) -> Optional[float]:
        a_rs = self.a_scaled(period_days)
        e = self.eccentricity
        w = math.radians(self.omega_deg)
        denom = max(1.0 - e**2, 1e-12)
        cos_i = (b / a_rs) * (1.0 + e * math.sin(w)) / denom
        if not np.isfinite(cos_i) or abs(cos_i) > 1.0:
            return None
        return float(np.degrees(np.arccos(cos_i)))

    def transit_duration_days(
        self,
        rp_rs: Optional[float] = None,
        inclination_deg: Optional[float] = None,
        period_days: Optional[float] = None,
    ) -> float:
        """
        Approximate first-to-fourth-contact duration for eccentric orbits.
        Used for masking/planning, not as a precision inference equation.
        """
        k = self.rp_rs if rp_rs is None else float(rp_rs)
        inc_deg = self.inclination_deg if inclination_deg is None else float(inclination_deg)
        P = self.period_days if period_days is None else float(period_days)
        inc = math.radians(inc_deg)
        a_rs = self.a_scaled(P)
        b = self.impact_parameter(inc_deg, P)

        if b >= 1.0 + k or math.sin(inc) <= 0:
            return 0.0

        e = self.eccentricity
        w = math.radians(self.omega_deg)
        geom = max((1.0 + k) ** 2 - b**2, 0.0)
        eccentric_factor = math.sqrt(max(1.0 - e**2, 1e-12)) / max(1.0 + e * math.sin(w), 1e-12)
        arg = (math.sqrt(geom) / (a_rs * math.sin(inc))) * eccentric_factor
        arg = float(np.clip(arg, -1.0, 1.0))
        return float((P / np.pi) * np.arcsin(arg))

    def _params(
        self,
        t0: float,
        period_days: float,
        rp_rs: float,
        inclination_deg: float,
        fp: float = 0.0,
    ) -> batman.TransitParams:
        params = batman.TransitParams()
        params.t0 = float(t0)
        params.per = float(period_days)
        params.rp = float(rp_rs)
        params.a = self.a_scaled(period_days)
        params.inc = float(inclination_deg)
        params.ecc = float(self.eccentricity)
        params.w = float(self.omega_deg)
        params.limb_dark = "quadratic"
        params.u = [float(self.u1), float(self.u2)]
        params.fp = float(fp)
        params.t_secondary = float(t0 + 0.5 * period_days)
        return params

    def _batman_model(
        self,
        params: batman.TransitParams,
        t: np.ndarray,
        transittype: str = "primary",
    ) -> batman.TransitModel:
        kwargs: Dict[str, Any] = {"transittype": transittype}
        if self.exposure_time_days > 0 and self.supersample_factor > 1:
            kwargs["supersample_factor"] = int(self.supersample_factor)
            kwargs["exp_time"] = float(self.exposure_time_days)
        return batman.TransitModel(params, np.asarray(t, dtype=float), **kwargs)

    @staticmethod
    def epoch_ttv_offsets(
        t: np.ndarray,
        t0: float,
        period_days: float,
        amplitude_minutes: float,
        ttv_period_days: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Piecewise-constant per-epoch timing shifts. Every datum associated with
        a given orbital epoch receives the same timing offset.
        """
        t = np.asarray(t, dtype=float)
        if amplitude_minutes <= 0 or ttv_period_days <= 0:
            epochs = np.rint((t - t0) / period_days).astype(int)
            return epochs, np.zeros_like(t)

        epochs = np.rint((t - t0) / period_days).astype(int)
        linear_tc = t0 + epochs * period_days
        amp_days = amplitude_minutes / 1440.0
        offsets = amp_days * np.sin(2.0 * np.pi * (linear_tc - t0) / ttv_period_days)
        return epochs, offsets

    def secondary_time(
        self,
        t0: float,
        period_days: float,
        rp_rs: Optional[float] = None,
        inclination_deg: Optional[float] = None,
    ) -> float:
        rp = self.rp_rs if rp_rs is None else float(rp_rs)
        inc = self.inclination_deg if inclination_deg is None else float(inclination_deg)
        params = self._params(t0, period_days, rp, inc, fp=max(self.thermal_fp, 1e-12))
        dummy_t = np.array([t0], dtype=float)
        model = self._batman_model(params, dummy_t, transittype="primary")
        return float(model.get_t_secondary(params))

    def generate_light_curve(
        self,
        t: np.ndarray,
        t0: float,
        period_days: Optional[float] = None,
        rp_rs: Optional[float] = None,
        inclination_deg: Optional[float] = None,
        ttv_amp_minutes: float = 0.0,
        ttv_period_days: float = 10.0,
        include_thermal_phase: bool = True,
    ) -> np.ndarray:
        """
        Full forward model:
          stellar primary transit
          + optional thermal companion phase flux
          + secondary occultation visibility
          + optional epoch-wise TTV shifts

        Planet flux is included exactly once, avoiding the previous
        secondary-eclipse/phase-curve double counting.
        """
        t = np.asarray(t, dtype=float)
        P = self.period_days if period_days is None else float(period_days)
        rp = self.rp_rs if rp_rs is None else float(rp_rs)
        inc = self.inclination_deg if inclination_deg is None else float(inclination_deg)

        _, offsets = self.epoch_ttv_offsets(t, t0, P, ttv_amp_minutes, ttv_period_days)
        t_warp = t - offsets

        fp = self.thermal_fp if include_thermal_phase else 0.0
        params = self._params(t0, P, rp, inc, fp=max(fp, 1e-16))

        # Apply epoch-wise TTV offsets to the primary transit only. This avoids
        # a discontinuous timing warp near secondary eclipse while preserving
        # the intended transit-timing perturbation.
        primary_model = self._batman_model(params, t_warp, transittype="primary")
        stellar_flux = primary_model.light_curve(params)

        if not include_thermal_phase or fp <= 0:
            return np.asarray(stellar_flux, dtype=float)

        # Thermal phase and secondary occultation remain on the linear orbital
        # ephemeris unless a dedicated eclipse-timing model is introduced.
        linear_primary_model = self._batman_model(params, t, transittype="primary")
        params.t_secondary = float(linear_primary_model.get_t_secondary(params))
        secondary_model = self._batman_model(params, t, transittype="secondary")
        secondary_flux = secondary_model.light_curve(params)

        # BATMAN secondary model is 1+fp outside occultation and ~1 in occultation.
        visibility = np.clip((secondary_flux - 1.0) / fp, 0.0, 1.0)

        phase_angle = 2.0 * np.pi * (t - params.t_secondary) / P
        day_fraction = 0.5 * (1.0 + np.cos(phase_angle))
        phase_fraction = self.nightside_fraction + (1.0 - self.nightside_fraction) * day_fraction
        planet_flux = fp * phase_fraction

        return np.asarray(stellar_flux + planet_flux * visibility, dtype=float)

    def primary_model_from_b(
        self,
        t: np.ndarray,
        rp_rs: float,
        impact_b: float,
        t0: float,
        period_days: float,
    ) -> np.ndarray:
        """Stateless primary-transit model for Bayesian inference."""
        inc = self.inclination_from_b(impact_b, period_days)
        if inc is None or impact_b < 0 or impact_b >= 1.0 + rp_rs:
            return np.full_like(np.asarray(t, dtype=float), np.nan)

        params = self._params(t0, period_days, rp_rs, inc, fp=0.0)
        model = self._batman_model(params, np.asarray(t, dtype=float), transittype="primary")
        return np.asarray(model.light_curve(params), dtype=float)


# =============================================================================
# 3. OBSERVATORY / NOISE / ETC
# =============================================================================

class ObservatoryProfiles:
    @staticmethod
    def get_profile(name: str) -> Dict[str, float]:
        # These are illustrative defaults, not authoritative instrument ETC values.
        profiles = {
            "MMT / MMIRS (Ground)": {
                "cadence_min": 5.0,
                "white_noise_ppm": 800.0,
                "red_noise_ppm": 1200.0,
                "night_hours": 9.0,
                "diameter_m": 6.5,
                "systematic_floor_ppm": 500.0,
            },
            "JWST / NIRSpec (Space)": {
                "cadence_min": 2.0,
                "white_noise_ppm": 100.0,
                "red_noise_ppm": 50.0,
                "night_hours": 24.0,
                "diameter_m": 6.5,
                "systematic_floor_ppm": 50.0,
            },
            "Nancy Grace Roman WFI (Space)": {
                "cadence_min": 2.0,
                "white_noise_ppm": 300.0,
                "red_noise_ppm": 200.0,
                "night_hours": 24.0,
                "diameter_m": 2.4,
                "systematic_floor_ppm": 100.0,
            },
            "Custom Observatory": {
                "cadence_min": 5.0,
                "white_noise_ppm": 1000.0,
                "red_noise_ppm": 500.0,
                "night_hours": 24.0,
                "diameter_m": 1.0,
                "systematic_floor_ppm": 500.0,
            },
        }
        return profiles[name].copy()


def simulate_red_noise(t: np.ndarray, sigma: float, tau_days: float, seed: int = 1234) -> np.ndarray:
    """Simple irregular-cadence AR(1)-like correlated-noise process."""
    t = np.asarray(t, dtype=float)
    if len(t) == 0 or sigma <= 0:
        return np.zeros_like(t)

    rng = np.random.default_rng(seed)
    x = np.zeros_like(t, dtype=float)
    x[0] = rng.normal(0.0, sigma)

    for i in range(1, len(t)):
        dt = max(t[i] - t[i - 1], 0.0)
        rho = math.exp(-dt / max(tau_days, 1e-6))
        x[i] = rho * x[i - 1] + sigma * math.sqrt(max(1.0 - rho**2, 0.0)) * rng.normal()

    return x


def simulate_astrophysical_variability(
    t: np.ndarray,
    amplitude: float,
    rotation_period_days: float,
) -> np.ndarray:
    if amplitude <= 0 or rotation_period_days <= 0:
        return np.zeros_like(t, dtype=float)
    p = rotation_period_days
    return amplitude * (
        np.sin(2.0 * np.pi * t / p)
        + 0.30 * np.sin(4.0 * np.pi * t / p + 0.7)
    )


class ApproximateETC:
    """
    Approximate photon-budget ETC.

    Assumptions:
    - input band magnitude is converted to AB
    - f_nu is treated as constant across a top-hat band
    - source, sky, dark, read noise are included
    - an irreducible systematic floor is added in quadrature
    - equal-duration out-of-transit baseline is assumed for a depth measurement
    """

    @staticmethod
    def source_electron_rate(
        magnitude: float,
        magnitude_system: str,
        diameter_m: float,
        throughput: float,
        bandpass: str,
    ) -> float:
        cfg = BANDS[bandpass]
        m_ab = float(magnitude)
        if magnitude_system == "Vega":
            m_ab += cfg["vega_to_ab"]

        # AB definition: 3631 Jy.
        f_nu = 3631.0e-26 * 10.0 ** (-0.4 * m_ab)  # W m^-2 Hz^-1

        center_m = cfg["center_um"] * 1e-6
        width_m = cfg["width_um"] * 1e-6
        lam_min = max(center_m - width_m / 2.0, 1e-9)
        lam_max = center_m + width_m / 2.0
        nu_max = const.c.value / lam_min
        nu_min = const.c.value / lam_max
        delta_nu = nu_max - nu_min

        energy_flux = f_nu * delta_nu
        photon_energy = const.h.value * const.c.value / center_m
        photon_rate_m2 = energy_flux / photon_energy

        area = np.pi * (diameter_m / 2.0) ** 2
        return float(photon_rate_m2 * area * np.clip(throughput, 0.0, 1.0))

    @staticmethod
    def transit_snr(
        magnitude: float,
        magnitude_system: str,
        diameter_m: float,
        throughput: float,
        bandpass: str,
        transit_depth: float,
        in_transit_hours: float,
        exposure_s: float,
        dead_time_s: float,
        aperture_pixels: int,
        sky_e_per_s_pix: float,
        dark_e_per_s_pix: float,
        read_noise_e: float,
        systematic_floor_ppm: float,
    ) -> Dict[str, float]:
        source_rate = ApproximateETC.source_electron_rate(
            magnitude, magnitude_system, diameter_m, throughput, bandpass
        )

        cadence_s = max(exposure_s + dead_time_s, 1e-6)
        n_exp = max(int((in_transit_hours * 3600.0) / cadence_s), 1)
        source_e = source_rate * exposure_s * n_exp

        sky_e = sky_e_per_s_pix * exposure_s * aperture_pixels * n_exp
        dark_e = dark_e_per_s_pix * exposure_s * aperture_pixels * n_exp
        read_var = (read_noise_e**2) * aperture_pixels * n_exp

        variance_in = source_e + sky_e + dark_e + read_var

        if source_e <= 0:
            return {
                "snr": 0.0,
                "source_e": 0.0,
                "n_exp": float(n_exp),
                "random_ppm": np.inf,
                "total_ppm": np.inf,
                "source_e_per_exp": 0.0,
            }

        # Depth compares in-transit to an equal-S/N out-of-transit baseline.
        relative_random = math.sqrt(2.0 * variance_in) / source_e
        floor = max(systematic_floor_ppm, 0.0) * 1e-6
        relative_total = math.sqrt(relative_random**2 + floor**2)
        snr = transit_depth / relative_total if relative_total > 0 else 0.0

        return {
            "snr": float(snr),
            "source_e": float(source_e),
            "n_exp": float(n_exp),
            "random_ppm": float(relative_random * 1e6),
            "total_ppm": float(relative_total * 1e6),
            "source_e_per_exp": float(source_rate * exposure_s),
        }


# =============================================================================
# 4. DATA QUALITY CONTROL
# =============================================================================

def combine_duplicate_times(t: np.ndarray, flux: np.ndarray, err: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"t": t, "flux": flux, "err": err})
    if not df["t"].duplicated().any():
        return t, flux, err

    rows = []
    for time_value, g in df.groupby("t", sort=True):
        e = g["err"].to_numpy(dtype=float)
        f = g["flux"].to_numpy(dtype=float)
        w = 1.0 / np.maximum(e, 1e-12) ** 2
        f_mean = np.sum(w * f) / np.sum(w)
        e_mean = math.sqrt(1.0 / np.sum(w))
        rows.append((float(time_value), float(f_mean), float(e_mean)))

    out = np.asarray(rows, dtype=float)
    return out[:, 0], out[:, 1], out[:, 2]


def prepare_lightcurve(
    t: np.ndarray,
    flux: np.ndarray,
    err: Optional[np.ndarray],
    clip_positive_spikes: bool = False,
    spike_sigma: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    report = {
        "rows_input": int(len(t)),
        "nonfinite_removed": 0,
        "invalid_error_removed": 0,
        "duplicates_combined": 0,
        "positive_spikes_removed": 0,
        "rows_retained": 0,
    }

    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err_arr = None if err is None else np.asarray(err, dtype=float)

    finite = np.isfinite(t) & np.isfinite(flux)
    if err_arr is not None:
        finite &= np.isfinite(err_arr)

    report["nonfinite_removed"] = int(len(t) - np.sum(finite))
    t = t[finite]
    flux = flux[finite]
    if err_arr is not None:
        err_arr = err_arr[finite]

    if len(t) < 10:
        raise ValueError("Too few valid data points after removing non-finite values.")

    med_flux = float(np.median(flux))
    if not np.isfinite(med_flux) or med_flux == 0:
        raise ValueError("Flux median is zero or invalid; VORTEX expects a positive photometric flux scale.")

    flux = flux / med_flux

    if err_arr is None:
        diffs = np.diff(flux)
        sigma = robust_sigma(diffs) / math.sqrt(2.0)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(float(np.std(flux)), 1e-6)
        err_arr = np.full_like(flux, sigma, dtype=float)
    else:
        err_arr = np.abs(err_arr / med_flux)
        valid_err = np.isfinite(err_arr) & (err_arr > 0)
        report["invalid_error_removed"] = int(len(err_arr) - np.sum(valid_err))
        t, flux, err_arr = t[valid_err], flux[valid_err], err_arr[valid_err]

    order = np.argsort(t)
    t, flux, err_arr = t[order], flux[order], err_arr[order]

    before_duplicates = len(t)
    t, flux, err_arr = combine_duplicate_times(t, flux, err_arr)
    report["duplicates_combined"] = int(before_duplicates - len(t))

    if clip_positive_spikes and len(t) >= 15:
        window = min(51, len(t) if len(t) % 2 == 1 else len(t) - 1)
        window = max(window, 5)
        trend = pd.Series(flux).rolling(window=window, center=True, min_periods=1).median().to_numpy()
        residual = flux - trend
        sig = robust_sigma(residual)
        if np.isfinite(sig) and sig > 0:
            keep = residual < spike_sigma * sig  # preserve negative transit-like dips
            report["positive_spikes_removed"] = int(np.sum(~keep))
            t, flux, err_arr = t[keep], flux[keep], err_arr[keep]

    report["rows_retained"] = int(len(t))
    return t, flux, err_arr, report


# =============================================================================
# 5. DETRENDING
# =============================================================================

def rolling_preliminary_detrend(
    t: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    window_days: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust first-pass detrend for blind search; does not use a transit ephemeris."""
    t = np.asarray(t, dtype=float)
    if len(t) < 5:
        return flux.copy(), np.ones_like(flux), err.copy()

    cadence = np.median(np.diff(t))
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = window_days / 25.0

    n_window = int(max(11, round(window_days / cadence)))
    n_window = min(n_window, 501)
    if n_window % 2 == 0:
        n_window += 1

    trend = pd.Series(flux).rolling(window=n_window, center=True, min_periods=1).median().to_numpy()
    trend = np.where(np.abs(trend) < 1e-8, 1.0, trend)
    cleaned = flux / trend
    cleaned_err = err / np.abs(trend)
    return cleaned, trend, cleaned_err


def gp_detrend_transit_aware(
    t: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    max_gp_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Fit GP only to out-of-transit data and use per-point measurement variance.
    Returns cleaned flux, multiplicative trend, cleaned errors, kernel string.
    """
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)

    phase = phase_from_ephemeris(t, t0, period)
    mask_half_width = max(1.5 * duration, 0.01 * period)
    oot = np.abs(phase) > mask_half_width

    if np.sum(oot) < max(20, int(0.25 * len(t))):
        # Safe fallback: do not pretend the GP fit is reliable.
        cleaned, trend, cleaned_err = rolling_preliminary_detrend(t, flux, err, window_days=max(0.5 * period, 0.2))
        return cleaned, trend, cleaned_err, "Rolling-median fallback (insufficient OOT points)"

    t_fit = t[oot]
    y_fit = flux[oot] - 1.0
    err_fit = err[oot]

    if len(t_fit) > max_gp_points:
        idx = np.linspace(0, len(t_fit) - 1, max_gp_points).astype(int)
        t_fit = t_fit[idx]
        y_fit = y_fit[idx]
        err_fit = err_fit[idx]

    baseline = max(float(np.ptp(t)), 1e-3)
    cadence = np.median(np.diff(np.sort(t)))
    cadence = max(float(cadence) if np.isfinite(cadence) and cadence > 0 else 1e-3, 1e-5)

    variance = max(float(np.var(y_fit)), 1e-10)
    ls0 = np.clip(0.2 * baseline, 5.0 * cadence, baseline)

    kernel = (
        ConstantKernel(variance, (1e-12, 1.0))
        * RBF(length_scale=ls0, length_scale_bounds=(max(2.0 * cadence, 1e-5), max(5.0 * baseline, 0.1)))
        + WhiteKernel(noise_level=max(float(np.median(err_fit) ** 2), 1e-12), noise_level_bounds=(1e-14, 1e-2))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.maximum(err_fit, 1e-10) ** 2,
        n_restarts_optimizer=1,
        normalize_y=False,
        random_state=42,
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(t_fit.reshape(-1, 1), y_fit)

    mu = gp.predict(t.reshape(-1, 1))
    trend = 1.0 + mu
    trend = np.where(np.abs(trend) < 1e-6, 1.0, trend)

    cleaned = flux / trend
    cleaned_err = err / np.abs(trend)
    return cleaned, trend, cleaned_err, str(gp.kernel_)


# =============================================================================
# 6. TLS
# =============================================================================

def tls_result_to_dict(results: Any) -> Dict[str, Any]:
    keys = [
        "period", "period_uncertainty", "T0", "duration", "depth", "rp_rs",
        "SDE", "SDE_raw", "FAP", "snr", "odd_even_mismatch",
        "transit_count", "distinct_transit_count",
        "periods", "power", "power_raw",
        "transit_times", "transit_depths", "transit_depths_uncertainties",
        "model_lightcurve_time", "model_lightcurve_model",
        "folded_phase", "folded_y", "folded_dy",
        "model_folded_phase", "model_folded_model",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        out[key] = getattr(results, key, np.nan)
    return out


@st.cache_data(show_spinner=False)
def cached_tls_search(
    t_tuple: Tuple[float, ...],
    f_tuple: Tuple[float, ...],
    e_tuple: Tuple[float, ...],
    host_mass_solar: float,
    host_radius_solar: float,
    period_min: float,
    period_max: float,
) -> Dict[str, Any]:
    t = np.asarray(t_tuple, dtype=float)
    flux = np.asarray(f_tuple, dtype=float)
    err = np.asarray(e_tuple, dtype=float)

    model = transitleastsquares(t, flux, err, verbose=False)

    # Widen allowed stellar ranges so brown-dwarf hosts remain valid.
    m_star = max(float(host_mass_solar), 0.0101)
    r_star = max(float(host_radius_solar), 0.1001)
    m_min = max(0.0101, 0.7 * m_star)
    m_max = max(m_min + 0.001, 1.3 * m_star)
    r_min = max(0.1001, 0.7 * r_star)
    r_max = max(r_min + 0.001, 1.3 * r_star)

    results = model.power(
        R_star=r_star,
        R_star_min=r_min,
        R_star_max=r_max,
        M_star=m_star,
        M_star_min=m_min,
        M_star_max=m_max,
        period_min=float(period_min),
        period_max=float(period_max),
        n_transits_min=2,
        oversampling_factor=3,
        show_progress_bar=False,
        use_threads=1,
    )
    return tls_result_to_dict(results)


def run_tls(
    t: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    engine: TransitEngine,
    period_min: float,
    period_max: float,
) -> Dict[str, Any]:
    return cached_tls_search(
        tuple(np.asarray(t, dtype=float)),
        tuple(np.asarray(flux, dtype=float)),
        tuple(np.asarray(err, dtype=float)),
        engine.host_mass_solar,
        engine.host_radius_solar,
        float(period_min),
        float(period_max),
    )


def _count_transits_in_baseline(t: np.ndarray, t0: float, period: float) -> int:
    if not np.isfinite(t0) or not np.isfinite(period) or period <= 0:
        return 0
    tmin, tmax = float(np.min(t)), float(np.max(t))
    kmin = int(math.floor((tmin - t0) / period)) - 1
    kmax = int(math.ceil((tmax - t0) / period)) + 1
    times = t0 + np.arange(kmin, kmax + 1) * period
    return int(np.sum((times >= tmin) & (times <= tmax)))


@st.cache_data(show_spinner=False)
def cached_bls_search(
    t_tuple: Tuple[float, ...],
    f_tuple: Tuple[float, ...],
    e_tuple: Tuple[float, ...],
    period_min: float,
    period_max: float,
) -> Dict[str, Any]:
    """Robust short-baseline fallback using Astropy Box Least Squares."""
    t = np.asarray(t_tuple, dtype=float)
    flux = np.asarray(f_tuple, dtype=float)
    err = np.asarray(e_tuple, dtype=float)

    span = float(np.ptp(t))
    cadence = float(np.median(np.diff(np.sort(t)))) if len(t) > 1 else np.nan
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = max(span / max(len(t) - 1, 1), 1e-4)

    pmin = max(float(period_min), 2.5 * cadence)
    pmax = min(float(period_max), span / 2.0)
    if pmax <= pmin:
        raise ValueError(
            f"The {span:.4f}-day baseline is too short for a blind periodic search over "
            f"{period_min:.4f}-{period_max:.4f} d while requiring at least two events."
        )

    dur_min = max(1.25 * cadence, 0.002, 0.01 * pmin)
    dur_max = min(max(4.0 * cadence, 2.0 * dur_min), 0.30 * pmin, 0.12 * span)
    if dur_max <= dur_min:
        dur_min = max(cadence, 0.005 * pmin)
        dur_max = min(0.45 * pmin, max(1.5 * dur_min, dur_min + cadence))
    if dur_max <= dur_min or dur_min >= pmin:
        raise ValueError("Cadence is too coarse relative to the requested minimum period for a meaningful BLS transit search.")
    durations = np.linspace(dur_min, dur_max, 6)

    model = BoxLeastSquares(t, flux, dy=np.maximum(err, 1e-12))
    results = model.autopower(
        durations,
        objective="snr",
        minimum_n_transit=2,
        minimum_period=pmin,
        maximum_period=pmax,
        frequency_factor=1.0,
    )

    power = np.asarray(results.power, dtype=float)
    if power.size == 0 or not np.any(np.isfinite(power)):
        raise ValueError("BLS could not construct a valid period grid for this short dataset.")
    idx = int(np.nanargmax(power))

    best_period = float(np.asarray(results.period, dtype=float)[idx])
    best_duration = float(np.asarray(results.duration, dtype=float)[idx])
    best_t0 = float(np.asarray(results.transit_time, dtype=float)[idx])
    depth_delta = float(np.asarray(results.depth, dtype=float)[idx])
    depth_snr = float(np.asarray(results.depth_snr, dtype=float)[idx])
    rp_rs = math.sqrt(max(depth_delta, 0.0))
    n_transits = _count_transits_in_baseline(t, best_t0, best_period)

    return {
        "period": best_period,
        "period_uncertainty": np.nan,
        "T0": best_t0,
        "duration": best_duration,
        "depth": 1.0 - depth_delta,  # mimic TLS convention used elsewhere in VORTEX
        "rp_rs": rp_rs,
        "SDE": np.nan,
        "SDE_raw": np.nan,
        "FAP": np.nan,
        "snr": depth_snr,
        "odd_even_mismatch": np.nan,
        "transit_count": n_transits,
        "distinct_transit_count": n_transits,
        "periods": np.asarray(results.period, dtype=float),
        "power": power,
        "power_raw": power,
        "search_method": "BLS (short-baseline fallback)",
        "search_note": "Astropy BLS was used because TLS is fragile for very short time series / very small trial grids.",
    }


def run_period_search(
    t: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    engine: TransitEngine,
    period_min: float,
    period_max: float,
) -> Dict[str, Any]:
    """Use TLS for suitable baselines and BLS as a safe short-baseline/error fallback."""
    span = float(np.ptp(t))
    if span < 2.0:
        return cached_bls_search(
            tuple(np.asarray(t, dtype=float)),
            tuple(np.asarray(flux, dtype=float)),
            tuple(np.asarray(err, dtype=float)),
            float(period_min),
            float(period_max),
        )

    try:
        result = run_tls(t, flux, err, engine, period_min, period_max)
        result["search_method"] = "TLS"
        result["search_note"] = "Transit Least Squares"
        return result
    except Exception as exc:
        # TLS can raise low-level NumPy/Index errors for pathological short or
        # tiny period grids. Never expose those as a Streamlit traceback; use
        # the robust BLS fallback instead.
        result = cached_bls_search(
            tuple(np.asarray(t, dtype=float)),
            tuple(np.asarray(flux, dtype=float)),
            tuple(np.asarray(err, dtype=float)),
            float(period_min),
            min(float(period_max), span / 2.0),
        )
        result["search_note"] = f"TLS failed ({exc}); VORTEX safely fell back to Astropy BLS."
        return result


def known_ephemeris_result(
    t: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    rp_rs: float,
) -> Dict[str, Any]:
    """Result shim when the baseline cannot identify a period but a known ephemeris is supplied."""
    return {
        "period": float(period),
        "period_uncertainty": np.nan,
        "T0": float(t0),
        "duration": float(duration),
        "depth": float(1.0 - rp_rs**2),
        "rp_rs": float(rp_rs),
        "SDE": np.nan,
        "SDE_raw": np.nan,
        "FAP": np.nan,
        "snr": np.nan,
        "odd_even_mismatch": np.nan,
        "transit_count": _count_transits_in_baseline(t, t0, period),
        "distinct_transit_count": _count_transits_in_baseline(t, t0, period),
        "periods": np.asarray([], dtype=float),
        "power": np.asarray([], dtype=float),
        "power_raw": np.asarray([], dtype=float),
        "search_method": "Known ephemeris (period not recoverable from baseline)",
        "search_note": "The dataset is shorter than two reference periods, so VORTEX did not claim an independent period recovery.",
    }


# =============================================================================
# 7. MCMC
# =============================================================================

@dataclass(frozen=True)
class MCMCBounds:
    rp_min: float
    rp_max: float
    b_min: float
    b_max: float
    t0_min: float
    t0_max: float
    p_min: float
    p_max: float
    log_jitter_min: float = math.log(1e-8)
    log_jitter_max: float = math.log(0.05)


def log_posterior(
    theta: np.ndarray,
    t: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    engine: TransitEngine,
    bounds: MCMCBounds,
) -> float:
    """
    Stateless posterior. The TransitEngine is frozen and never mutated.
    Parameters: Rp/Rs, impact parameter b, T0, period, log(jitter).
    """
    rp_rs, b, t0, period, log_jitter = theta

    if not (bounds.rp_min < rp_rs < bounds.rp_max):
        return -np.inf
    if not (bounds.b_min <= b < min(bounds.b_max, 1.0 + rp_rs)):
        return -np.inf
    if not (bounds.t0_min < t0 < bounds.t0_max):
        return -np.inf
    if not (bounds.p_min < period < bounds.p_max):
        return -np.inf
    if not (bounds.log_jitter_min < log_jitter < bounds.log_jitter_max):
        return -np.inf

    model = engine.primary_model_from_b(t, rp_rs, b, t0, period)
    if np.any(~np.isfinite(model)):
        return -np.inf

    jitter = math.exp(log_jitter)
    sigma2 = np.maximum(err, 1e-12) ** 2 + jitter**2
    residual = flux - model

    return float(-0.5 * np.sum(residual**2 / sigma2 + np.log(2.0 * np.pi * sigma2)))


def build_mcmc_bounds(tls: Dict[str, Any], engine: TransitEngine) -> MCMCBounds:
    p0 = float(tls["period"])
    t00 = float(tls["T0"])
    duration = max(float(tls["duration"]), 1e-4)
    p_unc = float(tls.get("period_uncertainty", np.nan))

    if not np.isfinite(p_unc) or p_unc <= 0:
        p_unc = 0.002 * p0

    p_half = max(5.0 * p_unc, 0.02 * p0)
    t0_half = max(2.5 * duration, 0.02 * p0)

    return MCMCBounds(
        rp_min=0.001,
        rp_max=1.5,
        b_min=0.0,
        b_max=2.0,
        t0_min=t00 - t0_half,
        t0_max=t00 + t0_half,
        p_min=max(1e-4, p0 - p_half),
        p_max=p0 + p_half,
    )


def initialize_walkers(
    n_walkers: int,
    tls: Dict[str, Any],
    engine: TransitEngine,
    bounds: MCMCBounds,
    err: np.ndarray,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    rp0 = float(tls.get("rp_rs", np.nan))
    if not np.isfinite(rp0) or rp0 <= 0:
        depth = float(tls.get("depth", 0.99))
        # TLS depth is a flux level at transit bottom, not simply delta.
        rp0 = math.sqrt(max(1.0 - depth, 1e-6))

    rp0 = float(np.clip(rp0, bounds.rp_min * 2, min(0.95 * bounds.rp_max, 1.2)))

    try:
        b0 = engine.impact_parameter()
    except Exception:
        b0 = 0.5
    if not np.isfinite(b0) or b0 < 0 or b0 >= 1.0 + rp0:
        b0 = min(0.5, 0.8 * (1.0 + rp0))

    t00 = float(tls["T0"])
    p0 = float(tls["period"])
    duration = max(float(tls["duration"]), 1e-4)
    p_unc = float(tls.get("period_uncertainty", np.nan))
    if not np.isfinite(p_unc) or p_unc <= 0:
        p_unc = max(1e-5, 1e-4 * p0)

    jitter0 = max(float(np.median(err)) * 0.5, 1e-7)
    center = np.array([rp0, b0, t00, p0, math.log(jitter0)])

    scales = np.array([
        max(0.03 * rp0, 5e-4),
        0.05,
        max(0.10 * duration, 1e-5),
        max(0.5 * p_unc, 1e-6),
        0.25,
    ])

    pos = np.empty((n_walkers, 5), dtype=float)
    for i in range(n_walkers):
        for _ in range(10000):
            trial = center + scales * rng.normal(size=5)
            if (
                bounds.rp_min < trial[0] < bounds.rp_max
                and bounds.b_min <= trial[1] < min(bounds.b_max, 1.0 + trial[0])
                and bounds.t0_min < trial[2] < bounds.t0_max
                and bounds.p_min < trial[3] < bounds.p_max
                and bounds.log_jitter_min < trial[4] < bounds.log_jitter_max
            ):
                pos[i] = trial
                break
        else:
            raise RuntimeError("Could not initialize MCMC walkers inside the priors.")

    return pos


def posterior_summary(samples: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        vals = samples[:, i]
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        rows.append({
            "Parameter": name,
            "Median": q50,
            "-1σ": q50 - q16,
            "+1σ": q84 - q50,
        })
    return pd.DataFrame(rows)


def make_trace_plot(chain: np.ndarray, names: list[str]) -> plt.Figure:
    n_steps, n_walkers, n_dim = chain.shape
    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2.0 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.plot(chain[:, :, d], alpha=0.25, lw=0.6)
        ax.set_ylabel(names[d])
    axes[-1].set_xlabel("MCMC step")
    fig.tight_layout()
    return fig


def make_corner_plot(
    samples: np.ndarray,
    names: list[str],
    truth: Optional[Dict[str, float]] = None,
    max_points: int = 8000,
) -> plt.Figure:
    """Dependency-free corner-style posterior plot using Matplotlib."""
    samples = np.asarray(samples, dtype=float)
    n_dim = len(names)
    if samples.shape[1] != n_dim:
        raise ValueError("Sample dimension does not match parameter names.")

    if len(samples) > max_points:
        idx = np.linspace(0, len(samples) - 1, max_points).astype(int)
        plot_samples = samples[idx]
    else:
        plot_samples = samples

    fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.5 * n_dim, 2.5 * n_dim))

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                ax.hist(plot_samples[:, j], bins=40, histtype="step", density=True)
                if truth and names[j] in truth:
                    ax.axvline(truth[names[j]], ls="--", lw=1.2)
            else:
                ax.scatter(plot_samples[:, j], plot_samples[:, i], s=2, alpha=0.12, rasterized=True)
                if truth:
                    if names[j] in truth:
                        ax.axvline(truth[names[j]], ls="--", lw=0.8)
                    if names[i] in truth:
                        ax.axhline(truth[names[i]], ls="--", lw=0.8)

            if i == n_dim - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            elif j > 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig


# =============================================================================
# 8. PLOTTING / BINNING
# =============================================================================

def weighted_phase_bins(
    t: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    t0: float,
    period: float,
    n_bins: int = 40,
) -> pd.DataFrame:
    phase = phase_from_ephemeris(t, t0, period)
    edges = np.linspace(-0.5 * period, 0.5 * period, n_bins + 1)
    rows = []

    for i in range(n_bins):
        m = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.sum(m) < 2:
            continue
        w = 1.0 / np.maximum(err[m], 1e-12) ** 2
        f = np.sum(w * flux[m]) / np.sum(w)
        e = math.sqrt(1.0 / np.sum(w))
        rows.append((0.5 * (edges[i] + edges[i + 1]), f, e, int(np.sum(m))))

    return pd.DataFrame(rows, columns=["Phase_Days", "Flux", "Error", "N"])


def make_ttv_table(
    t_min: float,
    t_max: float,
    t0: float,
    period: float,
    amp_minutes: float,
    ttv_period_days: float,
) -> pd.DataFrame:
    n0 = int(np.floor((t_min - t0) / period)) - 1
    n1 = int(np.ceil((t_max - t0) / period)) + 1
    epochs = np.arange(n0, n1 + 1)
    linear_tc = t0 + epochs * period

    if amp_minutes > 0 and ttv_period_days > 0:
        offsets_min = amp_minutes * np.sin(2.0 * np.pi * (linear_tc - t0) / ttv_period_days)
    else:
        offsets_min = np.zeros_like(linear_tc)

    actual_tc = linear_tc + offsets_min / 1440.0
    keep = (actual_tc >= t_min - period) & (actual_tc <= t_max + period)

    return pd.DataFrame({
        "Epoch": epochs[keep],
        "Linear_Tc": linear_tc[keep],
        "TTV_Minutes": offsets_min[keep],
        "Actual_Tc": actual_tc[keep],
    })


# =============================================================================
# 9. STREAMLIT PAGE / CSS
# =============================================================================

st.set_page_config(
    page_title="VORTEX v2.0 — Transit Exploration Suite",
    page_icon="🔭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #05080e; }
    section[data-testid="stSidebar"] { background-color: #090e18; }
    div[data-testid="stMetric"] {
        background-color: #0b111e;
        border: 1px solid #1a273e;
        border-radius: 6px;
        padding: 12px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔭 VORTEX v2.0.2")
st.caption("Visual Observatory for Retrieval & Transit Exploration — research prototype")

for key, default in {
    "mcmc_samples": None,
    "mcmc_log_prob": None,
    "mcmc_chain": None,
    "mcmc_truth": None,
    "mcmc_diag": None,
    "analysis_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# 10. SIDEBAR — DATA INPUT FIRST
# =============================================================================

st.sidebar.header("📂 Observational Data")
uploaded_file = st.sidebar.file_uploader("Upload reduced light curve (CSV)", type=["csv"])

uploaded_df: Optional[pd.DataFrame] = None
uploaded_time_internal: Optional[np.ndarray] = None
uploaded_flux: Optional[np.ndarray] = None
uploaded_err: Optional[np.ndarray] = None
time_offset = 0.0
time_label = "Elapsed Days"
time_is_absolute = False
qc_report: Optional[Dict[str, int]] = None

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
        st.stop()

    if len(uploaded_df.columns) < 2:
        st.sidebar.error("CSV must contain at least a time column and a flux column.")
        st.stop()

    columns = list(uploaded_df.columns)
    st.sidebar.subheader("📊 Map Data Columns")

    # Keep the original VORTEX upload philosophy: the astronomer explicitly
    # chooses which columns are time, flux, and uncertainty. We only provide
    # sensible defaults; we do not silently reinterpret the file.
    time_col = st.sidebar.selectbox("Time Array", columns, index=0)
    flux_col = st.sidebar.selectbox(
        "Flux Array",
        columns,
        index=default_flux_column_index(columns),
        help="CorrectedScienceFlux / normalized science flux is preferred when available.",
    )
    if time_col == flux_col:
        st.sidebar.error("Time Array and Flux Array must be different columns.")
        st.stop()

    err_choices = ["Estimate from robust scatter"] + columns
    likely_err = next(
        (i + 1 for i, c in enumerate(columns)
         if any(k in str(c).lower() for k in ["error", "err", "uncert", "sigma"])),
        0,
    )
    err_col_choice = st.sidebar.selectbox("Error Array", err_choices, index=likely_err)
    if err_col_choice in {time_col, flux_col}:
        st.sidebar.error("Error Array must be different from the selected Time/Flux columns.")
        st.stop()

    # Quality flag is also explicit. If the file contains OutlierRemoved (as
    # the user's MMIRS light curves do), select and honor it automatically.
    flag_candidates = [
        c for c in columns
        if any(k in str(c).strip().lower().replace(" ", "_")
               for k in ["outlier", "quality", "bad", "flag", "reject", "remove"])
    ]
    flag_choices = ["Do not use a quality flag"] + columns
    default_flag_index = 0
    preferred_flag = next(
        (c for c in columns
         if str(c).strip().lower().replace(" ", "_") in {
             "outlierremoved", "outlier_removed", "bad_row", "quality_flag"
         }),
        flag_candidates[0] if flag_candidates else None,
    )
    if preferred_flag is not None:
        default_flag_index = flag_choices.index(preferred_flag)

    quality_flag_choice = st.sidebar.selectbox(
        "Quality / outlier flag",
        flag_choices,
        index=default_flag_index,
        help="For boolean or numeric flags, True/non-zero is treated as a bad row and excluded.",
    )
    use_outlier_flag = quality_flag_choice != "Do not use a quality flag"
    outlier_flag_col = quality_flag_choice if use_outlier_flag else None

    st.sidebar.subheader("⏱ Time Interpretation")
    suggested_time_format = guess_time_format_from_column(
        time_col, uploaded_df[time_col].to_numpy()
    )
    time_format = suggested_time_format
    st.sidebar.caption(
        f"Detected from '{time_col}': **{suggested_time_format}**. "
        "Use Advanced time handling only if that interpretation is wrong."
    )

    # Frame cadence stays visible because it directly sets the physical time
    # scale of frame-index light curves and was part of the original workflow.
    frame_cadence_min = 5.0
    if time_format == "Frame Number":
        frame_cadence_min = st.sidebar.number_input(
            "Frame cadence (minutes)",
            min_value=0.001,
            value=5.0,
            step=0.1,
            help="Set this to the actual time between consecutive light-curve points.",
        )

    convert_to_bjd = False
    target_coord = None
    obs_location = None

    with st.sidebar.expander("Advanced time handling", expanded=False):
        override_time = st.checkbox(
            "Override automatic time interpretation",
            value=False,
            key="override_uploaded_time_format",
        )
        time_options = [
            "Frame Number", "Elapsed Minutes", "Elapsed Hours", "Elapsed Days",
            "BJD_TDB", "JD_UTC", "MJD_UTC",
        ]
        if override_time:
            time_format = st.selectbox(
                "Interpret selected time column as",
                time_options,
                index=time_options.index(suggested_time_format),
                key="uploaded_time_format_override",
            )
            if time_format == "Frame Number":
                frame_cadence_min = st.number_input(
                    "Frame cadence for override (minutes)",
                    min_value=0.001,
                    value=float(frame_cadence_min),
                    step=0.1,
                    key="frame_cadence_override",
                )

        if time_format in {"JD_UTC", "MJD_UTC"}:
            convert_to_bjd = st.checkbox(
                "Convert UTC times to BJD_TDB",
                value=False,
                help="Optional precision-timing conversion. Requires target and observatory coordinates.",
            )
            if convert_to_bjd:
                ra_text = st.text_input("Target RA (deg or hh:mm:ss)", value="0.0")
                dec_text = st.text_input("Target Dec (deg or dd:mm:ss)", value="0.0")
                obs_lat = st.number_input("Observatory latitude (deg)", value=0.0, format="%.6f")
                obs_lon = st.number_input("Observatory longitude (deg, east +)", value=0.0, format="%.6f")
                obs_height = st.number_input("Observatory elevation (m)", value=0.0, step=10.0)

                try:
                    target_coord = parse_skycoord(ra_text, dec_text)
                    obs_location = EarthLocation.from_geodetic(
                        lon=obs_lon * u.deg,
                        lat=obs_lat * u.deg,
                        height=obs_height * u.m,
                    )
                except Exception as exc:
                    st.error(f"Invalid coordinate input: {exc}")
                    st.stop()

    try:
        raw_t = uploaded_df[time_col].to_numpy()
        internal_t, time_offset, time_label, time_is_absolute = to_internal_time(
            raw_t,
            time_format,
            frame_cadence_min=frame_cadence_min,
            convert_utc_to_bjd=convert_to_bjd,
            coord=target_coord,
            location=obs_location,
        )
    except Exception as exc:
        st.sidebar.error(f"Time conversion failed: {exc}")
        st.stop()

    raw_flux = pd.to_numeric(uploaded_df[flux_col], errors="coerce").to_numpy(dtype=float)
    raw_err = None
    if err_col_choice != "Estimate from robust scatter":
        raw_err = pd.to_numeric(uploaded_df[err_col_choice], errors="coerce").to_numpy(dtype=float)

    flagged_rows_removed = 0
    if use_outlier_flag and outlier_flag_col is not None:
        bad_rows = boolean_bad_row_mask(uploaded_df[outlier_flag_col])
        if len(bad_rows) == len(internal_t):
            flagged_rows_removed = int(np.sum(bad_rows))
            keep_rows = ~bad_rows
            internal_t = internal_t[keep_rows]
            raw_flux = raw_flux[keep_rows]
            if raw_err is not None:
                raw_err = raw_err[keep_rows]

    clip_spikes = st.sidebar.checkbox("Clip obvious positive spikes", value=False)
    spike_sigma = st.sidebar.number_input("Positive-spike threshold (robust σ)", min_value=4.0, value=8.0, step=1.0)

    try:
        uploaded_time_internal, uploaded_flux, uploaded_err, qc_report = prepare_lightcurve(
            internal_t,
            raw_flux,
            raw_err,
            clip_positive_spikes=clip_spikes,
            spike_sigma=spike_sigma,
        )
        qc_report["flagged_rows_removed"] = int(flagged_rows_removed)
        qc_report["rows_input_original"] = int(len(uploaded_df))
    except Exception as exc:
        st.sidebar.error(f"Data quality-control failed: {exc}")
        st.stop()

    upload_span_days = float(np.ptp(uploaded_time_internal)) if len(uploaded_time_internal) > 1 else 0.0
    cadence_upload_min = (
        float(np.median(np.diff(uploaded_time_internal))) * 1440.0
        if len(uploaded_time_internal) > 1 else np.nan
    )
    st.sidebar.success(
        f"Loaded {len(uploaded_time_internal)} usable points | "
        f"baseline {upload_span_days:.4f} d ({24.0 * upload_span_days:.2f} h)"
    )
    if np.isfinite(cadence_upload_min):
        st.sidebar.caption(f"Median cadence after QC: {cadence_upload_min:.3f} min")

    if time_format in {"JD_UTC", "MJD_UTC"} and not convert_to_bjd:
        st.sidebar.warning("UTC times are being analyzed without barycentric correction. Do not use these T0 values for precision TTV work.")


# =============================================================================
# 11. SIDEBAR — SYSTEM / PHYSICS
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.header("🎯 Target System")

preset = st.sidebar.selectbox(
    "System preset",
    [
        "VHS 1256 b + Hypothetical Transiting Satellite",
        "TRAPPIST-1 c",
        "HD 209458 b",
        "Custom System",
    ],
)

if preset.startswith("VHS 1256"):
    defaults = dict(hm=19.0, hr=1.20, cr=2.51, p=0.80, inc=89.2, e=0.0, w=90.0, ht=1240, ct=500)
elif preset == "TRAPPIST-1 c":
    defaults = dict(
        hm=0.0898 * MSUN_TO_MJUP,
        hr=0.1192 * RSUN_TO_RJUP,
        cr=1.10,
        p=2.421937,
        inc=89.78,
        e=0.0,
        w=90.0,
        ht=2520,
        ct=500,
    )
elif preset == "HD 209458 b":
    defaults = dict(
        hm=1.10 * MSUN_TO_MJUP,
        hr=1.15 * RSUN_TO_RJUP,
        cr=15.58,
        p=3.524749,
        inc=86.7,
        e=0.0,
        w=90.0,
        ht=6070,
        ct=1450,
    )
else:
    defaults = dict(hm=10.0, hr=1.0, cr=2.0, p=1.5, inc=89.0, e=0.0, w=90.0, ht=3000, ct=1000)

st.sidebar.subheader("Physical Parameters")
hm = st.sidebar.number_input("Host mass (M_Jup)", min_value=0.01, value=float(defaults["hm"]), step=0.5)
hr = st.sidebar.number_input("Host radius (R_Jup)", min_value=0.05, value=float(defaults["hr"]), step=0.05)
cr = st.sidebar.number_input("Companion radius (R_Earth)", min_value=0.01, value=float(defaults["cr"]), step=0.1)
period_input = st.sidebar.number_input("Reference / injected period (days)", min_value=0.01, value=float(defaults["p"]), step=0.01, format="%.8f")
inc_input = st.sidebar.slider("Inclination (deg)", 70.0, 90.0, float(defaults["inc"]), 0.01)
ecc = st.sidebar.slider("Eccentricity", 0.0, 0.90, float(defaults["e"]), 0.01)
omega = st.sidebar.slider("Argument of periastron ω (deg)", 0.0, 360.0, float(defaults["w"]), 1.0)

bandpass = st.sidebar.selectbox("Bandpass", list(BANDS.keys()), index=0)

st.sidebar.subheader("Limb Darkening")
ld_mode = st.sidebar.selectbox("Quadratic LD coefficients", ["Approximate band defaults", "Custom"])
if ld_mode == "Approximate band defaults":
    u1, u2 = APPROX_LD[bandpass]
    st.sidebar.caption(f"Approximate coefficients: u1={u1:.3f}, u2={u2:.3f}. For publication work, replace with atmosphere/throughput-derived coefficients.")
else:
    u1 = st.sidebar.number_input("u1", value=float(APPROX_LD[bandpass][0]), step=0.01)
    u2 = st.sidebar.number_input("u2", value=float(APPROX_LD[bandpass][1]), step=0.01)
    if not quadratic_ld_is_physical(u1, u2):
        st.sidebar.warning("These quadratic limb-darkening coefficients may be non-physical.")

st.sidebar.subheader("Thermal / Timing")
host_temp = st.sidebar.number_input("Host effective temperature (K)", min_value=100.0, value=float(defaults["ht"]), step=50.0)
comp_temp = st.sidebar.number_input("Companion effective temperature (K)", min_value=50.0, value=float(defaults["ct"]), step=50.0)
include_thermal = st.sidebar.checkbox("Include approximate thermal phase + secondary eclipse", value=True)
nightside_fraction = st.sidebar.slider("Nightside / dayside thermal fraction", 0.0, 1.0, 0.0, 0.05)
ttv_amp = st.sidebar.number_input("Injected TTV amplitude (minutes)", min_value=0.0, value=0.0, step=0.5)
ttv_period = st.sidebar.number_input("Injected TTV period (days)", min_value=0.1, value=10.0, step=0.5)

st.sidebar.subheader("Exposure Integration")
exposure_seconds = st.sidebar.number_input("Exposure time (seconds)", min_value=0.0, value=60.0, step=10.0)
supersample = st.sidebar.selectbox("Transit-model supersampling", [1, 3, 5, 7, 9], index=2)

engine = TransitEngine(
    host_mass_jup=float(hm),
    host_radius_jup=float(hr),
    companion_radius_earth=float(cr),
    period_days=float(period_input),
    inclination_deg=float(inc_input),
    eccentricity=float(ecc),
    omega_deg=float(omega),
    host_temp_k=float(host_temp),
    companion_temp_k=float(comp_temp),
    bandpass=bandpass,
    u1=float(u1),
    u2=float(u2),
    exposure_time_days=float(exposure_seconds) / DAY_S,
    supersample_factor=int(supersample),
    nightside_fraction=float(nightside_fraction),
)

if engine.impact_parameter() >= 1.0 + engine.rp_rs:
    st.sidebar.error("Current geometry does not produce a transit (impact parameter is too large).")
    st.stop()


# =============================================================================
# 12. SIDEBAR — OBSERVATORY / SIMULATION / DETECTION
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.header("📡 Observatory / Simulation")
obs_name = st.sidebar.selectbox(
    "Observatory profile",
    ["MMT / MMIRS (Ground)", "JWST / NIRSpec (Space)", "Nancy Grace Roman WFI (Space)", "Custom Observatory"],
)
obs = ObservatoryProfiles.get_profile(obs_name)
st.sidebar.caption("Noise values below are illustrative defaults, not official instrument performance.")

cadence_min = st.sidebar.number_input("Cadence (minutes)", min_value=0.01, value=float(obs["cadence_min"]), step=0.1)
white_noise_ppm = st.sidebar.number_input("White noise per point (ppm)", min_value=0.0, value=float(obs["white_noise_ppm"]), step=50.0)
red_noise_ppm = st.sidebar.number_input("Correlated noise amplitude (ppm)", min_value=0.0, value=float(obs["red_noise_ppm"]), step=50.0)
red_tau_hours = st.sidebar.number_input("Correlated-noise timescale (hours)", min_value=0.01, value=1.0, step=0.1)
variability_amp = st.sidebar.number_input("Host variability amplitude (fraction)", min_value=0.0, value=0.0, step=0.001, format="%.4f")
rotation_period = st.sidebar.number_input("Host variability period (days)", min_value=0.01, value=0.35, step=0.05)

if uploaded_file is None:
    default_baseline = float(np.clip(3.0 * period_input, 3.0, 20.0))
    baseline = st.sidebar.slider("Synthetic baseline (days)", 1.0, 30.0, default_baseline, 0.5)
    simulation_t0 = 0.25 * period_input
    inject_signal = False
    detection_mode = "Known / Injection Recovery"
else:
    inject_signal = st.sidebar.checkbox("💉 Inject synthetic signal into uploaded data", value=False)
    detection_mode = st.sidebar.radio(
        "Detection mode",
        ["Blind Search", "Known / Injection Recovery"],
        index=0,
        help="Blind Search does not use the supplied period/T0 to mask the first detrending pass.",
    )
    baseline = float(np.ptp(uploaded_time_internal)) if uploaded_time_internal is not None else 1.0
    simulation_t0 = float(uploaded_time_internal.min() + 0.25 * period_input)

    if inject_signal:
        injection_t0_display = display_time(simulation_t0, time_offset)
        injection_t0_display = st.sidebar.number_input(
            f"Injected T0 ({time_label})",
            value=float(injection_t0_display),
            step=0.001,
            format="%.8f",
        )
        simulation_t0 = float(injection_t0_display - time_offset)

known_t0: Optional[float] = None
if uploaded_file is not None and detection_mode == "Known / Injection Recovery":
    default_known_display = display_time(simulation_t0, time_offset)
    known_t0_display = st.sidebar.number_input(
        f"Known/reference T0 ({time_label})",
        value=float(default_known_display),
        step=0.001,
        format="%.8f",
    )
    known_t0 = float(known_t0_display - time_offset)

if detection_mode == "Blind Search":
    span = max(float(np.ptp(uploaded_time_internal)), 0.1)
    default_pmax = min(max(2.0, period_input * 2.0), max(span / 2.0, 0.2))
    blind_pmin = st.sidebar.number_input("Blind-search P min (days)", min_value=0.05, value=0.10, step=0.05)
    blind_pmax = st.sidebar.number_input("Blind-search P max (days)", min_value=0.10, value=float(max(default_pmax, 0.2)), step=0.1)
    if blind_pmax <= blind_pmin:
        st.sidebar.error("Blind-search P max must exceed P min.")
        st.stop()
else:
    blind_pmin = max(0.05, 0.5 * period_input)
    blind_pmax = 2.0 * period_input


# =============================================================================
# 13. BUILD OBSERVED / SYNTHETIC LIGHT CURVE
# =============================================================================

if uploaded_file is None:
    dt = cadence_min / 1440.0
    t_full = np.arange(0.0, baseline, dt)

    if obs["night_hours"] < 24.0:
        mask = ((t_full * 24.0) % 24.0) <= obs["night_hours"]
        t_obs = t_full[mask]
    else:
        t_obs = t_full.copy()

    true_full_model = engine.generate_light_curve(
        t_full,
        t0=simulation_t0,
        period_days=period_input,
        ttv_amp_minutes=ttv_amp,
        ttv_period_days=ttv_period,
        include_thermal_phase=include_thermal,
    )
    true_obs_model = engine.generate_light_curve(
        t_obs,
        t0=simulation_t0,
        period_days=period_input,
        ttv_amp_minutes=ttv_amp,
        ttv_period_days=ttv_period,
        include_thermal_phase=include_thermal,
    )

    variability = simulate_astrophysical_variability(t_obs, variability_amp, rotation_period)
    red_noise = simulate_red_noise(t_obs, red_noise_ppm * 1e-6, red_tau_hours / 24.0, seed=1234)
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0.0, white_noise_ppm * 1e-6, len(t_obs))

    # Multiplicative astrophysical/correlated variability; additive photon/read-like white noise.
    raw_flux = true_obs_model * (1.0 + variability + red_noise) + white_noise
    raw_err = np.full_like(t_obs, max(white_noise_ppm * 1e-6, 1e-8), dtype=float)

    truth_available = True
    truth_t0 = simulation_t0
    truth_period = period_input
else:
    assert uploaded_time_internal is not None and uploaded_flux is not None and uploaded_err is not None
    t_obs = uploaded_time_internal.copy()
    raw_flux = uploaded_flux.copy()
    raw_err = uploaded_err.copy()
    t_full = t_obs.copy()
    true_full_model = np.ones_like(t_obs)

    if inject_signal:
        injected = engine.generate_light_curve(
            t_obs,
            t0=simulation_t0,
            period_days=period_input,
            ttv_amp_minutes=ttv_amp,
            ttv_period_days=ttv_period,
            include_thermal_phase=include_thermal,
        )
        raw_flux = raw_flux * injected  # physically multiplicative injection
        true_full_model = injected
        truth_available = True
        truth_t0 = simulation_t0
        truth_period = period_input
    else:
        truth_available = False
        truth_t0 = np.nan
        truth_period = np.nan


if len(t_obs) < 20:
    st.error("VORTEX needs at least 20 usable observations.")
    st.stop()

span_days = float(np.ptp(t_obs))
if span_days <= 0:
    st.error("Observation times have zero baseline.")
    st.stop()


# =============================================================================
# 14. DETECTION PIPELINE
# =============================================================================

try:
    with st.spinner("Running VORTEX detection pipeline..."):
        if detection_mode == "Blind Search":
            # First detrend without knowing transit period or T0.
            prelim_flux, prelim_trend, prelim_err = rolling_preliminary_detrend(
                t_obs, raw_flux, raw_err, window_days=max(0.25, min(1.0, 0.15 * span_days))
            )

            tls_first = run_period_search(
                t_obs,
                prelim_flux,
                prelim_err,
                engine,
                blind_pmin,
                min(blind_pmax, max(span_days / 2.0, blind_pmin * 1.01)),
            )

            first_period = float(tls_first["period"])
            first_t0 = float(tls_first["T0"])
            first_duration = float(tls_first["duration"])

            cleaned_flux, gp_trend, cleaned_err, gp_kernel = gp_detrend_transit_aware(
                t_obs, raw_flux, raw_err, first_period, first_t0, first_duration
            )

            tls_final = run_period_search(
                t_obs,
                cleaned_flux,
                cleaned_err,
                engine,
                blind_pmin,
                min(blind_pmax, max(span_days / 2.0, blind_pmin * 1.01)),
            )
        else:
            ephem_t0 = simulation_t0 if uploaded_file is None else float(known_t0)
            duration_for_mask = engine.transit_duration_days(period_days=period_input)
            if duration_for_mask <= 0:
                duration_for_mask = 0.05 * period_input

            cleaned_flux, gp_trend, cleaned_err, gp_kernel = gp_detrend_transit_aware(
                t_obs, raw_flux, raw_err, period_input, ephem_t0, duration_for_mask
            )

            search_pmin = max(0.05, 0.5 * period_input)
            search_pmax = min(2.0 * period_input, max(span_days / 2.0, search_pmin * 1.01))

            # A period cannot be independently recovered if fewer than ~2 cycles are covered.
            # In known/injection mode, use the supplied ephemeris rather than forcing TLS/BLS
            # to manufacture a periodogram from an underconstrained single-night dataset.
            if span_days < 2.0 * period_input:
                tls_final = known_ephemeris_result(
                    t_obs, cleaned_flux, period_input, ephem_t0, duration_for_mask, engine.rp_rs
                )
            else:
                tls_final = run_period_search(
                    t_obs,
                    cleaned_flux,
                    cleaned_err,
                    engine,
                    search_pmin,
                    search_pmax,
                )
except Exception as exc:
    st.error(
        "The period-search stage could not construct a meaningful search for this dataset. "
        f"{type(exc).__name__}: {exc}"
    )
    st.info(
        "Check the selected Time Array and its interpretation/cadence. For a single-night "
        "dataset with a known longer period, use 'Known / Injection Recovery' rather than a blind periodic search."
    )
    st.stop()

recovered_period = float(tls_final["period"])
recovered_t0 = float(tls_final["T0"])
recovered_duration = float(tls_final["duration"])
recovered_rp_rs = float(tls_final.get("rp_rs", np.nan))
recovered_sde = float(tls_final.get("SDE", np.nan))
recovered_fap = float(tls_final.get("FAP", np.nan))
recovered_snr = float(tls_final.get("snr", np.nan))

if not np.isfinite(recovered_period) or recovered_period <= 0:
    st.error("The period-search engine did not return a valid period.")
    st.stop()

if span_days < 2.0 * recovered_period:
    st.warning("The baseline contains fewer than ~2 recovered periods. Period recovery may be poorly constrained.")


# =============================================================================
# 15. RESET STALE MCMC RESULTS WHEN ANALYSIS CHANGES
# =============================================================================

signature_payload = np.array(
    [
        len(t_obs),
        np.nanmean(t_obs),
        np.nanstd(t_obs),
        np.nanmean(cleaned_flux),
        np.nanstd(cleaned_flux),
        recovered_period,
        recovered_t0,
        hm,
        hr,
        ecc,
        omega,
    ],
    dtype=float,
).tobytes()

analysis_signature = hashlib.sha1(signature_payload).hexdigest()
if st.session_state.analysis_signature != analysis_signature:
    st.session_state.analysis_signature = analysis_signature
    st.session_state.mcmc_samples = None
    st.session_state.mcmc_log_prob = None
    st.session_state.mcmc_chain = None
    st.session_state.mcmc_truth = None
    st.session_state.mcmc_diag = None


# =============================================================================
# 16. HUD
# =============================================================================

search_method = str(tls_final.get("search_method", "TLS"))
search_note = str(tls_final.get("search_note", ""))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Recovered / Reference Period", f"{recovered_period:.6f} d")
m2.metric("Search Method", search_method)
if np.isfinite(recovered_sde):
    m3.metric("TLS SDE", f"{recovered_sde:.2f}")
elif np.isfinite(recovered_snr):
    m3.metric("Detection SNR", f"{recovered_snr:.2f}")
else:
    m3.metric("Detection Score", "—")
m4.metric("Recovered / Reference T0", f"{display_time(recovered_t0, time_offset):.6f}")
m5.metric("Thermal Fp/F★", f"{engine.thermal_fp:.2e}")

if search_note:
    st.caption(search_note)
if np.isfinite(recovered_fap):
    st.caption(f"TLS white-noise FAP estimate: {recovered_fap:.3g}. Correlated real noise can make this estimate optimistic.")


# =============================================================================
# 17. TABS
# =============================================================================

tabs = st.tabs([
    "📈 Light Curve",
    "🔍 Period Search",
    "🌗 Phase / TTV",
    "🎲 Bayesian Retrieval",
    "📸 Approx. ETC",
    "🧪 Data / Export",
])

# -----------------------------------------------------------------------------
# LIGHT CURVE
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Observed and Detrended Light Curve")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t_obs + time_offset, raw_flux, ".", alpha=0.45, label="Raw / injected data")
    ax.plot(t_obs + time_offset, gp_trend, lw=1.8, label="GP / baseline trend")
    ax.plot(t_obs + time_offset, cleaned_flux, ".", ms=3, alpha=0.75, label="Detrended data")

    if truth_available:
        model_t = np.linspace(t_obs.min(), t_obs.max(), min(5000, max(1000, len(t_obs))))
        truth_model = engine.generate_light_curve(
            model_t,
            t0=truth_t0,
            period_days=truth_period,
            ttv_amp_minutes=ttv_amp,
            ttv_period_days=ttv_period,
            include_thermal_phase=include_thermal,
        )
        ax.plot(model_t + time_offset, truth_model, lw=1.5, ls="--", label="Injected / simulated truth")

    ax.set_xlabel(time_label)
    ax.set_ylabel("Normalized Flux")
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(f"Transit-aware GP kernel: {gp_kernel}")

# -----------------------------------------------------------------------------
# PERIOD SEARCH
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Transit Period Search")
    st.caption(f"Method: {search_method}")

    p_grid = np.asarray(tls_final["periods"], dtype=float)
    power = np.asarray(tls_final["power"], dtype=float)

    if p_grid.size > 0 and power.size > 0:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(p_grid, power, lw=1.2, label=f"{search_method} periodogram")
        ax.axvline(recovered_period, ls="--", lw=1.5, label="Recovered period")

        if truth_available:
            ax.axvline(truth_period, ls=":", lw=1.5, label="Injected truth")

        ax.set_xlabel("Trial Period (days)")
        ax.set_ylabel("Search power / score")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No independent periodogram is shown because the supplied/known period is longer than the baseline can independently recover.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Period", f"{recovered_period:.8f} d")
    c2.metric("T0", f"{display_time(recovered_t0, time_offset):.8f}")
    c3.metric("Duration", f"{24.0 * recovered_duration:.3f} hr")
    c4.metric("Rp/Rs", f"{recovered_rp_rs:.5f}" if np.isfinite(recovered_rp_rs) else "—")
    c5.metric("Distinct transits", f"{int(tls_final.get('distinct_transit_count', 0))}")

    stats_df = pd.DataFrame({
        "Statistic": ["SDE", "SNR", "FAP", "Period uncertainty (days)", "Odd-even mismatch (σ)"],
        "Value": [
            recovered_sde,
            recovered_snr,
            recovered_fap,
            float(tls_final.get("period_uncertainty", np.nan)),
            float(tls_final.get("odd_even_mismatch", np.nan)),
        ],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PHASE / TTV
# -----------------------------------------------------------------------------
with tabs[2]:
    left, right = st.columns(2)

    with left:
        st.subheader("Phase-Folded Transit")
        phase = phase_from_ephemeris(t_obs, recovered_t0, recovered_period)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(phase, cleaned_flux, yerr=cleaned_err, fmt=".", alpha=0.25, label="Detrended data")

        binned = weighted_phase_bins(
            t_obs, cleaned_flux, cleaned_err, recovered_t0, recovered_period, n_bins=50
        )
        if len(binned):
            ax.errorbar(
                binned["Phase_Days"],
                binned["Flux"],
                yerr=binned["Error"],
                fmt="o",
                ms=4,
                capsize=2,
                label="Weighted bins",
            )

        model_phase = np.linspace(-0.5 * recovered_period, 0.5 * recovered_period, 1500)
        rp_for_model = recovered_rp_rs if np.isfinite(recovered_rp_rs) else engine.rp_rs
        model_flux = engine.generate_light_curve(
            model_phase + recovered_t0,
            t0=recovered_t0,
            period_days=recovered_period,
            rp_rs=rp_for_model,
            inclination_deg=engine.inclination_deg,
            include_thermal_phase=False,
        )
        ax.plot(model_phase, model_flux, lw=1.8, label="Transit model")
        ax.set_xlabel("Time from mid-transit (days)")
        ax.set_ylabel("Normalized Flux")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with right:
        st.subheader("Thermal Phase / Secondary Eclipse")
        sec_time = engine.secondary_time(recovered_t0, recovered_period)
        phase_t = np.linspace(sec_time - 0.12 * recovered_period, sec_time + 0.12 * recovered_period, 800)
        phase_flux = engine.generate_light_curve(
            phase_t,
            t0=recovered_t0,
            period_days=recovered_period,
            include_thermal_phase=True,
        )
        fig_phase = go.Figure(
            go.Scatter(
                x=phase_t + time_offset,
                y=phase_flux,
                mode="lines",
                name="Thermal + eclipse model",
            )
        )
        fig_phase.update_layout(
            xaxis_title=time_label,
            yaxis_title="Relative Flux",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_phase, use_container_width=True)
        st.caption("Thermal curve uses blackbodies and a top-hat approximation to the selected bandpass.")

    st.markdown("---")
    st.subheader("Ephemeris")

    epochs = np.arange(0, 8)
    ephem_internal = recovered_t0 + epochs * recovered_period
    ephem = pd.DataFrame({
        "Epoch": epochs,
        f"Predicted Mid-Transit ({time_label})": ephem_internal + time_offset,
        "Observability": ["Requires RA/Dec, site coordinates, and a real observing date" for _ in epochs],
    })
    st.dataframe(ephem, use_container_width=True, hide_index=True)

    if truth_available and ttv_amp > 0:
        st.subheader("Injected Epoch-wise TTV O−C")
        ttv_df = make_ttv_table(
            t_obs.min(), t_obs.max(), truth_t0, truth_period, ttv_amp, ttv_period
        )
        fig_ttv = go.Figure(
            go.Scatter(
                x=ttv_df["Epoch"],
                y=ttv_df["TTV_Minutes"],
                mode="lines+markers",
                name="Injected TTV",
            )
        )
        fig_ttv.update_layout(
            xaxis_title="Epoch",
            yaxis_title="O−C (minutes)",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig_ttv, use_container_width=True)

# -----------------------------------------------------------------------------
# BAYESIAN RETRIEVAL
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Bayesian Transit Retrieval")
    st.caption("Samples Rp/Rs, impact parameter b, T0, period, and an additional white-jitter term. The likelihood is stateless.")

    if ttv_amp > 0 and truth_available:
        st.warning("The retrieval below assumes a linear ephemeris. Injected TTVs can broaden or bias the posterior unless TTV parameters are modeled explicitly.")

    n_walkers = st.selectbox("Walkers", [24, 32, 48, 64], index=1)
    n_steps = st.slider("MCMC steps", 500, 5000, 1200, 100)
    max_burn = max(100, n_steps - 100)
    default_burn = min(max(200, n_steps // 4), max_burn)
    burn_in = st.slider("Burn-in steps", 100, max_burn, default_burn, 50)
    thin = st.selectbox("Thinning", [1, 2, 5, 10], index=1)

    mcmc_bounds = build_mcmc_bounds(tls_final, engine)

    if st.button("🚀 Run MCMC Retrieval", type="primary"):
        pos = initialize_walkers(n_walkers, tls_final, engine, mcmc_bounds, cleaned_err)

        with st.spinner(f"Running {n_walkers} walkers × {n_steps} steps..."):
            sampler = emcee.EnsembleSampler(
                n_walkers,
                5,
                log_posterior,
                args=(t_obs, cleaned_flux, cleaned_err, engine, mcmc_bounds),
            )
            sampler.run_mcmc(pos, n_steps, progress=False)

        flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
        full_chain = sampler.get_chain()

        acceptance = float(np.mean(sampler.acceptance_fraction))
        tau = np.full(5, np.nan)
        try:
            tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
        except Exception:
            pass

        st.session_state.mcmc_samples = flat_samples
        st.session_state.mcmc_log_prob = flat_log_prob
        st.session_state.mcmc_chain = full_chain
        st.session_state.mcmc_diag = {
            "acceptance": acceptance,
            "tau": tau,
            "burn_in": burn_in,
            "thin": thin,
            "n_walkers": n_walkers,
            "n_steps": n_steps,
        }

        truth_dict = None
        if truth_available:
            true_b = engine.impact_parameter(period_days=truth_period)
            truth_dict = {
                "Rp/Rs": engine.rp_rs,
                "b": true_b,
                "T0": truth_t0,
                "Period": truth_period,
            }
        st.session_state.mcmc_truth = truth_dict

    if st.session_state.mcmc_samples is not None:
        names = ["Rp/Rs", "b", "T0", "Period", "log_jitter"]
        samples = np.asarray(st.session_state.mcmc_samples)
        logp = np.asarray(st.session_state.mcmc_log_prob)
        chain = np.asarray(st.session_state.mcmc_chain)
        diag = st.session_state.mcmc_diag

        d1, d2, d3 = st.columns(3)
        d1.metric("Mean acceptance", f"{diag['acceptance']:.3f}")
        finite_tau = np.asarray(diag["tau"], dtype=float)
        if np.any(np.isfinite(finite_tau)):
            d2.metric("Max autocorr time", f"{np.nanmax(finite_tau):.1f} steps")
            ess = samples.shape[0] / max(np.nanmax(finite_tau), 1.0)
            d3.metric("Approx. ESS", f"{ess:.0f}")
        else:
            d2.metric("Autocorr time", "Not reliable yet")
            d3.metric("Posterior samples", f"{len(samples):,}")

        summary = posterior_summary(samples, names)
        # Show T0 in the original absolute time convention when appropriate.
        t0_row = summary["Parameter"] == "T0"
        summary.loc[t0_row, "Median"] += time_offset
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Walker Traces")
        trace_fig = make_trace_plot(chain, names)
        st.pyplot(trace_fig)

        st.subheader("Posterior Contours")
        truth_for_plot = st.session_state.mcmc_truth
        corner_names = ["Rp/Rs", "b", "T0", "Period"]
        corner_samples = samples[:, :4]
        try:
            corner_fig = make_corner_plot(corner_samples, corner_names, truth_for_plot)
            st.pyplot(corner_fig)
        except Exception as exc:
            st.warning(f"Posterior contour plot unavailable: {exc}")

        posterior_export = pd.DataFrame(samples, columns=names)
        posterior_export["T0_Display"] = posterior_export["T0"] + time_offset
        posterior_export["log_posterior"] = logp
        st.download_button(
            "📥 Download Posterior Samples",
            posterior_export.to_csv(index=False),
            file_name="vortex_posterior_samples.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# ETC
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Approximate Exposure-Time / Transit-SNR Calculator")
    st.warning("This quick ETC is for planning and comparisons. Use an instrument-specific ETC for proposal/publication-grade predictions.")

    e1, e2, e3 = st.columns(3)
    target_mag = e1.number_input("Target magnitude", value=14.0, step=0.1)
    mag_system = e2.selectbox("Magnitude system", ["AB", "Vega"], index=1 if "Band" in bandpass else 0)
    diameter_m = e3.number_input("Telescope diameter (m)", min_value=0.05, value=float(obs["diameter_m"]), step=0.1)

    e4, e5, e6 = st.columns(3)
    throughput = e4.slider("Total throughput", 0.01, 1.0, 0.25, 0.01)
    etc_exposure = e5.number_input("Exposure time (s)", min_value=0.1, value=max(exposure_seconds, 1.0), step=1.0)
    dead_time = e6.number_input("Readout/dead time (s)", min_value=0.0, value=5.0, step=1.0)

    e7, e8, e9 = st.columns(3)
    aperture_pixels = int(e7.number_input("Photometric aperture pixels", min_value=1, value=50, step=1))
    sky_rate = e8.number_input("Sky/background (e⁻/s/pix)", min_value=0.0, value=5.0 if "Ground" in obs_name else 0.1, step=0.1)
    dark_rate = e9.number_input("Dark current (e⁻/s/pix)", min_value=0.0, value=0.01, step=0.01)

    e10, e11 = st.columns(2)
    read_noise = e10.number_input("Read noise (e⁻/pix/exposure)", min_value=0.0, value=10.0, step=0.5)
    floor_ppm = e11.number_input("Systematic floor (ppm)", min_value=0.0, value=float(obs["systematic_floor_ppm"]), step=10.0)

    transit_depth = engine.rp_rs**2
    default_hours = max(24.0 * recovered_duration, 0.5)
    in_hours = st.number_input("Total in-transit integration (hours)", min_value=0.01, value=float(default_hours), step=0.1)

    etc = ApproximateETC.transit_snr(
        magnitude=target_mag,
        magnitude_system=mag_system,
        diameter_m=diameter_m,
        throughput=throughput,
        bandpass=bandpass,
        transit_depth=transit_depth,
        in_transit_hours=in_hours,
        exposure_s=etc_exposure,
        dead_time_s=dead_time,
        aperture_pixels=aperture_pixels,
        sky_e_per_s_pix=sky_rate,
        dark_e_per_s_pix=dark_rate,
        read_noise_e=read_noise,
        systematic_floor_ppm=floor_ppm,
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Transit SNR", f"{etc['snr']:.2f} σ")
    q2.metric("Random precision", f"{etc['random_ppm']:.1f} ppm")
    q3.metric("Total precision", f"{etc['total_ppm']:.1f} ppm")
    q4.metric("N exposures", f"{int(etc['n_exp'])}")

    st.caption(f"Approximate source electrons per exposure: {etc['source_e_per_exp']:.3e}")

# -----------------------------------------------------------------------------
# DATA / EXPORT
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Data Quality / Reproducibility")

    if qc_report is not None:
        qc_df = pd.DataFrame({
            "Check": [
                "Rows input",
                "Rows excluded by quality/outlier flag",
                "Non-finite removed",
                "Invalid uncertainties removed",
                "Duplicate timestamps combined",
                "Positive spikes removed",
                "Rows retained",
            ],
            "Count": [
                qc_report.get("rows_input_original", qc_report["rows_input"]),
                qc_report.get("flagged_rows_removed", 0),
                qc_report["nonfinite_removed"],
                qc_report["invalid_error_removed"],
                qc_report["duplicates_combined"],
                qc_report["positive_spikes_removed"],
                qc_report["rows_retained"],
            ],
        })
        st.dataframe(qc_df, use_container_width=True, hide_index=True)
    else:
        st.info("Synthetic data mode: no uploaded-data QC report.")

    cadence_est = np.median(np.diff(t_obs)) * 1440.0
    qc_metrics = pd.DataFrame({
        "Metric": ["Baseline (days)", "Median cadence (min)", "Median raw flux", "Median uncertainty"],
        "Value": [span_days, cadence_est, np.median(raw_flux), np.median(raw_err)],
    })
    st.dataframe(qc_metrics, use_container_width=True, hide_index=True)

    export_df = pd.DataFrame({
        f"Time_{time_label}": t_obs + time_offset,
        "Time_Internal_Days": t_obs,
        "Raw_Flux": raw_flux,
        "Trend_Model": gp_trend,
        "Cleaned_Flux": cleaned_flux,
        "Raw_Error": raw_err,
        "Cleaned_Error": cleaned_err,
    })

    st.download_button(
        "📥 Download Reduced Light Curve",
        export_df.to_csv(index=False),
        file_name="vortex_reduced_lightcurve.csv",
        mime="text/csv",
    )

    tls_export = pd.DataFrame({
        "Period_Days": np.asarray(tls_final["periods"], dtype=float),
        "Search_Power": np.asarray(tls_final["power"], dtype=float),
    })
    st.download_button(
        "📥 Download Periodogram",
        tls_export.to_csv(index=False),
        file_name="vortex_periodogram.csv",
        mime="text/csv",
    )

    st.subheader("Software Versions")
    versions = pd.DataFrame({
        "Package": [
            "numpy", "pandas", "streamlit", "matplotlib", "plotly",
            "batman-package", "transitleastsquares", "emcee",
            "scikit-learn", "astropy",
        ],
        "Version": [
            safe_version("numpy"),
            safe_version("pandas"),
            safe_version("streamlit"),
            safe_version("matplotlib"),
            safe_version("plotly"),
            safe_version("batman-package"),
            safe_version("transitleastsquares"),
            safe_version("emcee"),
            safe_version("scikit-learn"),
            safe_version("astropy"),
        ],
    })
    st.dataframe(versions, use_container_width=True, hide_index=True)


# =============================================================================
# 18. FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align:center; margin-bottom:0;'>"
    "<strong>VORTEX v2.0.2</strong> | Designed &amp; Developed by <strong>Amanpreet Singh</strong><br>"
    "Astrophysics &amp; Transit Modeling | University of Arizona"
    "</p>",
    unsafe_allow_html=True,
)
st.caption(
    "VORTEX is designed for reproducible transit simulation and analysis. "
    "Thermal-band and ETC modules are explicitly approximate; validate against "
    "instrument-specific tools before publication/proposal use."
)
