"""
VORTEX Professional v3.0
Visual Observatory for Retrieval & Transit Exploration

Professional transit simulation and analysis workbench with synthetic and real-data
workflows, injection/recovery, TLS/BLS search, transit-aware Gaussian-process
detrending, Bayesian retrieval, observatory comparison, multi-night analysis,
archive/mission ingestion, observing planning, vetting, and reproducible export.

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
import io
import json
import math
import re
import certifi
import requests
import warnings
import zipfile
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import batman
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from transitleastsquares import transitleastsquares
from scipy.signal import find_peaks

# Optional professional integrations. VORTEX stays usable when these are absent.
try:
    from astroplan import (
        AirmassConstraint, AltitudeConstraint, AtNightConstraint, FixedTarget,
        MoonSeparationConstraint, Observer, observability_table,
    )
    ASTROPLAN_AVAILABLE = True
except Exception:
    ASTROPLAN_AVAILABLE = False

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except Exception:
    LIGHTKURVE_AVAILABLE = False

try:
    from exotic_ld import StellarLimbDarkening
    EXOTIC_LD_AVAILABLE = True
except Exception:
    EXOTIC_LD_AVAILABLE = False



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


def calc_throughput_flux_ratio(
    companion_temp_k: float,
    host_temp_k: float,
    companion_radius_m: float,
    host_radius_m: float,
    wavelength_um: Sequence[float],
    throughput: Sequence[float],
) -> float:
    """Blackbody flux ratio integrated through an arbitrary throughput curve.

    The throughput may be in any non-negative relative normalization because the
    same response is applied to host and companion. Wavelengths are in microns.
    """
    wav_um = np.asarray(wavelength_um, dtype=float)
    tr = np.asarray(throughput, dtype=float)
    good = np.isfinite(wav_um) & np.isfinite(tr) & (wav_um > 0) & (tr >= 0)
    wav_um, tr = wav_um[good], tr[good]
    if len(wav_um) < 3 or np.nanmax(tr) <= 0:
        raise ValueError("Throughput curve needs at least 3 positive-wavelength points and non-zero throughput.")
    order = np.argsort(wav_um)
    wav_m = wav_um[order] * 1e-6
    tr = tr[order]
    host = _trapz(planck_lambda(wav_m, host_temp_k) * tr, wav_m)
    comp = _trapz(planck_lambda(wav_m, companion_temp_k) * tr, wav_m)
    if host <= 0 or not np.isfinite(host) or not np.isfinite(comp):
        return 0.0
    return float((companion_radius_m / host_radius_m) ** 2 * comp / host)


# =============================================================================
# 2. TRANSIT ENGINE
# =============================================================================

@dataclass(frozen=True)
class TransitEngine:
    host_mass_jup: float
    host_radius_jup: float
    companion_radius_earth: float
    companion_mass_earth: float = 0.0
    period_days: float = 1.0
    inclination_deg: float = 89.0
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
    throughput_wavelength_um: Optional[Tuple[float, ...]] = None
    throughput_values: Optional[Tuple[float, ...]] = None

    @property
    def host_mass_kg(self) -> float:
        return float(self.host_mass_jup * const.M_jup.value)

    @property
    def host_radius_m(self) -> float:
        return float(self.host_radius_jup * const.R_jup.value)

    @property
    def companion_mass_kg(self) -> float:
        return float(max(self.companion_mass_earth, 0.0) * const.M_earth.value)

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
        if self.throughput_wavelength_um is not None and self.throughput_values is not None:
            return calc_throughput_flux_ratio(
                self.companion_temp_k, self.host_temp_k,
                self.companion_radius_m, self.host_radius_m,
                self.throughput_wavelength_um, self.throughput_values,
            )
        return calc_bandpass_flux_ratio(
            self.companion_temp_k,
            self.host_temp_k,
            self.companion_radius_m,
            self.host_radius_m,
            self.bandpass,
        )

    def a_scaled(self, period_days: Optional[float] = None) -> float:
        P = float(self.period_days if period_days is None else period_days) * DAY_S
        total_mass = self.host_mass_kg + self.companion_mass_kg
        a_m = (const.G.value * total_mass * P**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
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
    """Instrument/observatory profiles used for simulation defaults.

    Noise numbers are explicitly illustrative unless the user enters measured
    values. Site coordinates for MMT come from the MMT Observatory. Space
    profiles do not attempt to reproduce detailed mission pointing constraints.
    """

    PROFILES = {
        "MMT / MMIRS (Ground)": {
            "cadence_min": 5.0, "white_noise_ppm": 800.0, "red_noise_ppm": 1200.0,
            "night_hours": 9.0, "diameter_m": 6.5, "systematic_floor_ppm": 500.0,
            "space": False, "lat_deg": 31.6887778, "lon_deg": -110.8845556,
            "height_m": 2606.0, "band_min_um": 0.9, "band_max_um": 2.4,
            "coverage_note": "Idealized 9-hour nightly sampling with explicit daytime gaps for simulator comparisons; use Observing Planner for date-specific visibility.",
        },
        "Lazuli 3-m (Space; idealized continuous benchmark)": {
            "cadence_min": 5.0, "white_noise_ppm": 100.0, "red_noise_ppm": 50.0,
            "night_hours": 24.0, "diameter_m": 3.0, "systematic_floor_ppm": 50.0,
            "space": True, "lat_deg": np.nan, "lon_deg": np.nan, "height_m": np.nan,
            "band_min_um": 0.4, "band_max_um": 1.7,
            "coverage_note": "24-hour coverage is an idealized mathematical benchmark, not a published target-by-target Lazuli visibility model.",
        },
        "JWST / NIRSpec (Space)": {
            "cadence_min": 2.0, "white_noise_ppm": 100.0, "red_noise_ppm": 50.0,
            "night_hours": 24.0, "diameter_m": 6.5, "systematic_floor_ppm": 50.0,
            "space": True, "lat_deg": np.nan, "lon_deg": np.nan, "height_m": np.nan,
            "band_min_um": 0.6, "band_max_um": 5.3,
            "coverage_note": "Continuous coverage is an idealized simulator mode; real JWST visibility/operations constraints are not modeled here.",
        },
        "Nancy Grace Roman WFI (Space)": {
            "cadence_min": 2.0, "white_noise_ppm": 300.0, "red_noise_ppm": 200.0,
            "night_hours": 24.0, "diameter_m": 2.4, "systematic_floor_ppm": 100.0,
            "space": True, "lat_deg": np.nan, "lon_deg": np.nan, "height_m": np.nan,
            "band_min_um": 0.48, "band_max_um": 2.3,
            "coverage_note": "Idealized space-observatory sampling; detailed Roman operations constraints are not modeled.",
        },
        "Custom Ground Observatory": {
            "cadence_min": 5.0, "white_noise_ppm": 1000.0, "red_noise_ppm": 500.0,
            "night_hours": 9.0, "diameter_m": 1.0, "systematic_floor_ppm": 500.0,
            "space": False, "lat_deg": 0.0, "lon_deg": 0.0, "height_m": 0.0,
            "band_min_um": 0.3, "band_max_um": 5.0,
            "coverage_note": "User-defined ground profile.",
        },
        "Custom Space Observatory": {
            "cadence_min": 5.0, "white_noise_ppm": 300.0, "red_noise_ppm": 100.0,
            "night_hours": 24.0, "diameter_m": 1.0, "systematic_floor_ppm": 100.0,
            "space": True, "lat_deg": np.nan, "lon_deg": np.nan, "height_m": np.nan,
            "band_min_um": 0.3, "band_max_um": 5.0,
            "coverage_note": "User-defined idealized space profile.",
        },
    }

    @staticmethod
    def get_profile(name: str) -> Dict[str, Any]:
        if name not in ObservatoryProfiles.PROFILES:
            raise KeyError(f"Unknown observatory profile: {name}")
        return ObservatoryProfiles.PROFILES[name].copy()

    @staticmethod
    def names() -> List[str]:
        return list(ObservatoryProfiles.PROFILES.keys())


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
        # Per-point measurement variance is already supplied through alpha below.
        # WhiteKernel therefore represents only additional *unmodeled* white jitter.
        + WhiteKernel(noise_level=max(float((0.10 * np.median(err_fit)) ** 2), 1e-14), noise_level_bounds=(1e-14, 1e-2))
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
# 9. PROFESSIONAL INTEGRATIONS / SCIENCE DIAGNOSTICS
# =============================================================================

APP_VERSION = "3.0.0"

TARGET_PRESETS: Dict[str, Dict[str, float]] = {
    "VHS 1256 b + Hypothetical Transiting Satellite": {
        "host_mass_jup": 19.0,
        "host_radius_jup": 1.20,
        "companion_radius_earth": 2.51,
        "companion_mass_earth": 5.0,
        "period_days": 0.80,
        "inclination_deg": 89.2,
        "eccentricity": 0.0,
        "omega_deg": 90.0,
        "host_temp_k": 1240.0,
        "companion_temp_k": 300.0,
        "logg": 3.5,
        "metallicity": 0.0,
        "ra_deg": 194.0080,
        "dec_deg": -12.9564,
    },
    "TRAPPIST-1 c": {
        "host_mass_jup": 0.0898 * MSUN_TO_MJUP,
        "host_radius_jup": 0.1192 * RSUN_TO_RJUP,
        "companion_radius_earth": 1.10,
        "companion_mass_earth": 1.31,
        "period_days": 2.4218,
        "inclination_deg": 89.67,
        "eccentricity": 0.0,
        "omega_deg": 90.0,
        "host_temp_k": 2566.0,
        "companion_temp_k": 340.0,
        "logg": 5.24,
        "metallicity": 0.04,
        "ra_deg": 346.6224,
        "dec_deg": -5.0413,
    },
    "HD 209458 b": {
        "host_mass_jup": 1.148 * MSUN_TO_MJUP,
        "host_radius_jup": 1.155 * RSUN_TO_RJUP,
        "companion_radius_earth": 15.58,
        "companion_mass_earth": 232.0,
        "period_days": 3.5247486,
        "inclination_deg": 86.71,
        "eccentricity": 0.0,
        "omega_deg": 90.0,
        "host_temp_k": 6071.0,
        "companion_temp_k": 1450.0,
        "logg": 4.37,
        "metallicity": 0.0,
        "ra_deg": 330.7949,
        "dec_deg": 18.8842,
    },
}


def escape_adql_string(value: str) -> str:
    return str(value).replace("'", "''")


@st.cache_data(ttl=3600, show_spinner=False)
def query_nasa_exoplanet_archive(target_name: str) -> pd.DataFrame:
    """Query one target from the NASA Exoplanet Archive PSCompPars TAP table.

    HTTPS certificate verification remains enabled and explicitly uses Certifi's
    CA bundle.  This avoids the common macOS/Python virtual-environment problem
    where urllib/OpenSSL cannot find the system certificate chain.
    """
    target = escape_adql_string(target_name.strip())
    if not target:
        raise ValueError("Enter a planet or host name.")

    columns = [
        "pl_name", "hostname", "ra", "dec", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
        "pl_tranmid", "pl_tranmiderr1", "pl_tranmiderr2", "pl_rade", "pl_radeerr1", "pl_radeerr2",
        "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_orbeccen", "pl_orbincl", "pl_orblper",
        "st_mass", "st_masserr1", "st_masserr2", "st_rad", "st_raderr1", "st_raderr2",
        "st_teff", "st_tefferr1", "st_tefferr2", "st_logg", "st_met",
        "sy_vmag", "sy_jmag", "sy_hmag", "sy_kmag", "disc_year", "discoverymethod",
    ]
    where = (
        f"lower(pl_name)=lower('{target}') OR lower(hostname)=lower('{target}') "
        f"OR lower(pl_name) like lower('%{target}%')"
    )
    adql = f"select {','.join(columns)} from pscomppars where {where} order by pl_name"

    endpoint = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    request_params = {"query": adql, "format": "csv"}

    try:
        response = requests.get(
            endpoint,
            params=request_params,
            timeout=(10, 30),
            verify=certifi.where(),
            headers={
                "User-Agent": "VORTEX-Professional/3.0 (+NASA-Exoplanet-Archive-client)",
                "Accept": "text/csv, text/plain;q=0.9, */*;q=0.1",
            },
        )
        response.raise_for_status()
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(
            "NASA Exoplanet Archive SSL verification failed even with the Certifi CA bundle. "
            f"Certifi bundle: {certifi.where()}. Original error: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "NASA Exoplanet Archive query timed out. Check your internet connection and try again."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the NASA Exoplanet Archive. Check your network/VPN/proxy and try again. "
            f"Original error: {exc}"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:500] if exc.response is not None else ""
        raise RuntimeError(
            f"NASA Exoplanet Archive returned HTTP {status}. Response: {body}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"NASA Exoplanet Archive query failed: {exc}") from exc

    payload = response.text
    if not payload.strip():
        raise RuntimeError("NASA Exoplanet Archive returned an empty response.")

    try:
        df = pd.read_csv(io.StringIO(payload))
    except Exception as exc:
        preview = payload[:500].replace("\n", " ")
        raise RuntimeError(
            f"NASA Exoplanet Archive returned data that could not be parsed as CSV. Preview: {preview}"
        ) from exc

    if df.empty:
        raise ValueError(f"No PSCompPars match found for '{target_name}'.")
    return df


def archive_row_to_target(row: pd.Series) -> Dict[str, float]:
    def val(key: str, fallback: float) -> float:
        x = pd.to_numeric(pd.Series([row.get(key, np.nan)]), errors="coerce").iloc[0]
        return float(x) if np.isfinite(x) else float(fallback)

    st_mass = val("st_mass", 1.0)
    st_rad = val("st_rad", 1.0)
    pl_rade = val("pl_rade", 1.0)
    pl_mass = val("pl_bmasse", 0.0)
    return {
        "host_mass_jup": st_mass * MSUN_TO_MJUP,
        "host_radius_jup": st_rad * RSUN_TO_RJUP,
        "companion_radius_earth": pl_rade,
        "companion_mass_earth": pl_mass,
        "period_days": val("pl_orbper", 1.0),
        "inclination_deg": val("pl_orbincl", 89.0),
        "eccentricity": max(0.0, min(0.95, val("pl_orbeccen", 0.0))),
        "omega_deg": val("pl_orblper", 90.0) % 360.0,
        "host_temp_k": val("st_teff", 5500.0),
        "companion_temp_k": 1000.0,
        "logg": val("st_logg", 4.5),
        "metallicity": val("st_met", 0.0),
        "ra_deg": val("ra", 0.0),
        "dec_deg": val("dec", 0.0),
        "archive_t0_bjd": val("pl_tranmid", np.nan),
    }


def parse_throughput_upload(uploaded: Any) -> Tuple[Tuple[float, ...], Tuple[float, ...], pd.DataFrame]:
    if uploaded is None:
        raise ValueError("No throughput file supplied.")
    df = pd.read_csv(uploaded)
    if df.shape[1] < 2:
        raise ValueError("Throughput CSV needs at least two columns: wavelength and throughput.")
    w = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    tr = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(w) & np.isfinite(tr) & (w > 0) & (tr >= 0)
    w, tr = w[good], tr[good]
    if len(w) < 3:
        raise ValueError("Throughput file has fewer than three usable rows.")
    # Conservative unit heuristic: Angstrom > 100, nm > 10, micron otherwise.
    med = float(np.nanmedian(w))
    if med > 1000:
        w_um = w / 1e4
    elif med > 10:
        w_um = w / 1000.0
    else:
        w_um = w
    order = np.argsort(w_um)
    w_um, tr = w_um[order], tr[order]
    clean = pd.DataFrame({"Wavelength_um": w_um, "Throughput": tr})
    return tuple(w_um.tolist()), tuple(tr.tolist()), clean


@st.cache_data(ttl=86400, show_spinner=False)
def compute_exotic_ld_coefficients(
    metallicity: float,
    teff: float,
    logg: float,
    ld_model: str,
    mode: str,
    wav_start_angstrom: float,
    wav_end_angstrom: float,
    custom_wavelength_angstrom: Optional[Tuple[float, ...]] = None,
    custom_throughput: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, float]:
    if not EXOTIC_LD_AVAILABLE:
        raise RuntimeError("ExoTiC-LD is not installed. Install package 'exotic-ld'.")
    sld = StellarLimbDarkening(
        M_H=float(metallicity), Teff=float(teff), logg=float(logg),
        ld_model=ld_model, ld_data_path="exotic_ld_data", verbose=0,
    )
    kwargs: Dict[str, Any] = {}
    if mode == "custom":
        if custom_wavelength_angstrom is None or custom_throughput is None:
            raise ValueError("Custom ExoTiC-LD mode requires a throughput curve.")
        kwargs["custom_wavelengths"] = np.asarray(custom_wavelength_angstrom, dtype=float)
        kwargs["custom_throughput"] = np.asarray(custom_throughput, dtype=float)
    coeffs = sld.compute_quadratic_ld_coeffs(
        wavelength_range=[float(wav_start_angstrom), float(wav_end_angstrom)],
        mode=mode,
        **kwargs,
    )
    return float(coeffs[0]), float(coeffs[1])


def physical_system_checks(engine: TransitEngine) -> List[Tuple[str, str]]:
    messages: List[Tuple[str, str]] = []
    if engine.host_mass_jup <= 0 or engine.host_radius_jup <= 0 or engine.companion_radius_earth <= 0:
        messages.append(("error", "Masses/radii must be positive."))
        return messages
    if not (0 <= engine.eccentricity < 1):
        messages.append(("error", "Bound Keplerian orbit requires 0 ≤ eccentricity < 1."))
    if engine.period_days <= 0:
        messages.append(("error", "Orbital period must be positive."))
        return messages
    a_rs = engine.a_scaled()
    rp = engine.rp_rs
    if a_rs <= 1.0 + rp:
        messages.append(("error", f"Orbit intersects the bodies: a/Rhost={a_rs:.3f} ≤ 1+Rp/Rhost={1+rp:.3f}."))
    b = engine.impact_parameter()
    if b >= 1.0 + rp:
        messages.append(("warning", f"Geometry is non-transiting for the current parameters (b={b:.3f}, 1+Rp/R★={1+rp:.3f})."))
    if engine.companion_mass_earth > 0:
        rho_host = engine.host_mass_kg / (4.0 / 3.0 * np.pi * engine.host_radius_m**3)
        rho_comp = engine.companion_mass_kg / (4.0 / 3.0 * np.pi * engine.companion_radius_m**3)
        if rho_host > 0 and rho_comp > 0:
            # Fluid Roche limit; relevant as a warning, not a hard boundary for rigid bodies.
            roche_m = 2.44 * engine.host_radius_m * (rho_host / rho_comp) ** (1.0 / 3.0)
            a_m = a_rs * engine.host_radius_m
            if a_m < roche_m:
                messages.append(("warning", f"Semi-major axis lies inside the approximate fluid Roche limit ({a_m/engine.host_radius_m:.2f} vs {roche_m/engine.host_radius_m:.2f} host radii)."))
    return messages


def phase_coverage_fraction(t: np.ndarray, period: float, bins: int = 100) -> float:
    if len(t) == 0 or period <= 0:
        return 0.0
    phase = np.mod(np.asarray(t, dtype=float), period) / period
    hist, _ = np.histogram(phase, bins=bins, range=(0, 1))
    return float(np.mean(hist > 0))


def transit_coverage_metrics(t: np.ndarray, t0: float, period: float, duration: float) -> Dict[str, float]:
    t = np.asarray(t, dtype=float)
    if len(t) < 2 or period <= 0 or duration <= 0:
        return {"n_predicted": 0, "n_any": 0, "n_half": 0, "mean_fraction": 0.0}
    cadence = float(np.median(np.diff(np.sort(t))))
    cadence = max(cadence, 1e-8)
    kmin = int(np.floor((t.min() - t0) / period)) - 1
    kmax = int(np.ceil((t.max() - t0) / period)) + 1
    centers = t0 + np.arange(kmin, kmax + 1) * period
    centers = centers[(centers >= t.min() - 0.5 * duration) & (centers <= t.max() + 0.5 * duration)]
    fractions = []
    expected = max(int(np.ceil(duration / cadence)), 1)
    for tc in centers:
        n = int(np.sum(np.abs(t - tc) <= 0.5 * duration))
        fractions.append(min(n / expected, 1.0))
    arr = np.asarray(fractions, dtype=float)
    return {
        "n_predicted": int(len(arr)),
        "n_any": int(np.sum(arr > 0)),
        "n_half": int(np.sum(arr >= 0.5)),
        "mean_fraction": float(np.mean(arr)) if len(arr) else 0.0,
    }


def spectral_window(t: np.ndarray, period_min: float, period_max: float, n_freq: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    pmin = max(float(period_min), 1e-4)
    pmax = max(float(period_max), pmin * 1.01)
    freqs = np.linspace(1.0 / pmax, 1.0 / pmin, n_freq)
    centered = t - np.mean(t)
    # |sum exp(-2πift)|^2 / N^2 is a standard normalized sampling spectral window.
    phase = -2j * np.pi * freqs[:, None] * centered[None, :]
    power = np.abs(np.exp(phase).sum(axis=1)) ** 2 / max(len(t), 1) ** 2
    return 1.0 / freqs, power


def build_sampling_times(
    profile: Dict[str, Any],
    baseline_days: float,
    cadence_min: float,
    phase_offset_days: float = 0.0,
    apply_ground_daytime_gaps: bool = True,
    night_hours_override: Optional[float] = None,
) -> np.ndarray:
    """
    Build the observation timestamps used by the synthetic simulator.

    Space profiles are sampled continuously.

    Ground profiles can explicitly include the regular day/night window:
    only ``night_hours`` out of each 24-hour cycle are retained.  The
    remaining samples are not generated at all, so the daytime intervals
    are genuine gaps in the dataset rather than cosmetic gaps in a plot.

    ``phase_offset_days`` shifts the observing window relative to the
    synthetic orbital ephemeris, which is useful for testing favorable and
    unfavorable transit phases.
    """
    dt = float(cadence_min) / 1440.0
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cadence must be positive.")

    t = np.arange(0.0, float(baseline_days), dt)

    is_space = bool(profile.get("space", False))
    default_night_hours = float(profile.get("night_hours", 24.0))
    night_hours = (
        default_night_hours
        if night_hours_override is None
        else float(night_hours_override)
    )
    night_hours = float(np.clip(night_hours, 0.0, 24.0))

    # Space observatories, disabled gap modeling, or a 24-hour observing
    # window remain continuously sampled.
    if is_space or (not apply_ground_daytime_gaps) or night_hours >= 24.0:
        return t

    if night_hours <= 0:
        return np.asarray([], dtype=float)

    # The synthetic "night" occupies the first night_hours of each shifted
    # 24-hour cycle. The phase offset controls where that window lands.
    cycle_hour = np.mod((t + float(phase_offset_days)) * 24.0, 24.0)
    observed = cycle_hour < night_hours
    return t[observed]


def ground_daytime_intervals(
    baseline_days: float,
    night_hours: float,
    phase_offset_days: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Return exact daytime intervals for the idealized repeating ground window.

    These intervals match build_sampling_times(): each 24-hour cycle retains
    ``night_hours`` and rejects the rest as daylight/not observed.
    """
    baseline_days = float(baseline_days)
    night_hours = float(np.clip(night_hours, 0.0, 24.0))
    if baseline_days <= 0 or night_hours >= 24.0:
        return []

    # Work in an unshifted cycle coordinate u = t + phase_offset.
    u_start = float(phase_offset_days)
    u_end = baseline_days + float(phase_offset_days)

    first_day = int(np.floor(u_start)) - 1
    last_day = int(np.ceil(u_end)) + 1

    intervals: List[Tuple[float, float]] = []
    night_fraction = night_hours / 24.0

    for day in range(first_day, last_day + 1):
        daylight_u0 = day + night_fraction
        daylight_u1 = day + 1.0

        # Transform back to plotted elapsed time.
        x0 = daylight_u0 - float(phase_offset_days)
        x1 = daylight_u1 - float(phase_offset_days)

        x0 = max(0.0, x0)
        x1 = min(baseline_days, x1)
        if x1 > x0:
            intervals.append((float(x0), float(x1)))

    return intervals


def add_daytime_gap_shading(
    fig: go.Figure,
    baseline_days: float,
    night_hours: float,
    phase_offset_days: float = 0.0,
    x_offset: float = 0.0,
    label_first: bool = True,
) -> None:
    """Shade the exact idealized daytime intervals used by the simulator."""
    intervals = ground_daytime_intervals(
        baseline_days=baseline_days,
        night_hours=night_hours,
        phase_offset_days=phase_offset_days,
    )

    for i, (x0, x1) in enumerate(intervals):
        kwargs: Dict[str, Any] = {
            "x0": x0 + float(x_offset),
            "x1": x1 + float(x_offset),
            "fillcolor": "rgba(150, 150, 150, 0.22)",
            "line_width": 0,
            "layer": "below",
        }
        if label_first and i == 0:
            kwargs["annotation_text"] = "Daylight / not observed"
            kwargs["annotation_position"] = "top left"
        fig.add_vrect(**kwargs)


def break_lines_at_large_gaps(
    x: np.ndarray,
    y: np.ndarray,
    gap_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so a Plotly line does not bridge unobserved intervals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return x, y

    positive_dt = np.diff(x)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if len(positive_dt) == 0:
        return x, y

    cadence = float(np.median(positive_dt))
    threshold = max(float(gap_factor) * cadence, cadence + 1e-12)

    xs: List[float] = []
    ys: List[float] = []
    for i in range(len(x)):
        xs.append(float(x[i]))
        ys.append(float(y[i]))
        if i < len(x) - 1 and (x[i + 1] - x[i]) > threshold:
            xs.append(np.nan)
            ys.append(np.nan)

    return np.asarray(xs), np.asarray(ys)


def simulate_profile_observation(
    engine: TransitEngine,
    profile: Dict[str, Any],
    t0: float,
    baseline_days: float,
    cadence_min: float,
    white_ppm: float,
    red_ppm: float,
    red_tau_hours: float,
    seed: int,
    window_phase_days: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = build_sampling_times(profile, baseline_days, cadence_min, window_phase_days)
    model = engine.generate_light_curve(t, t0=t0, include_thermal_phase=False)
    red = simulate_red_noise(t, red_ppm * 1e-6, red_tau_hours / 24.0, seed=seed + 17)
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, white_ppm * 1e-6, len(t))
    flux = model * (1.0 + red) + white
    err = np.full_like(t, max(white_ppm * 1e-6, 1e-8), dtype=float)
    return t, flux, err, model


def fast_bls_candidate(
    t: np.ndarray, flux: np.ndarray, err: np.ndarray, pmin: float, pmax: float,
    expected_duration: Optional[float] = None,
) -> Dict[str, float]:
    t = np.asarray(t, dtype=float)
    span = float(np.ptp(t))
    cadence = max(float(np.median(np.diff(np.sort(t)))), 1e-5)
    pmin = max(float(pmin), 3.0 * cadence)
    pmax = min(float(pmax), max(span / 2.0, pmin * 1.02))
    if pmax <= pmin:
        return {"period": np.nan, "snr": np.nan, "duration": np.nan, "t0": np.nan, "depth": np.nan}
    if expected_duration is None or not np.isfinite(expected_duration) or expected_duration <= 0:
        d0 = max(2.0 * cadence, 0.02 * pmin)
    else:
        d0 = max(float(expected_duration), 2.0 * cadence)
    durations = np.unique(np.clip(np.array([0.6, 0.8, 1.0, 1.25, 1.5]) * d0, 1.5 * cadence, 0.25 * pmin))
    durations = durations[(durations > cadence) & (durations < pmin)]
    if len(durations) == 0:
        return {"period": np.nan, "snr": np.nan, "duration": np.nan, "t0": np.nan, "depth": np.nan}
    bls = BoxLeastSquares(t, flux, dy=np.maximum(err, 1e-12))
    r = bls.autopower(
        durations, objective="snr", minimum_n_transit=2,
        minimum_period=pmin, maximum_period=pmax, frequency_factor=2.0,
    )
    power = np.asarray(r.power, dtype=float)
    if power.size == 0 or not np.any(np.isfinite(power)):
        return {"period": np.nan, "snr": np.nan, "duration": np.nan, "t0": np.nan, "depth": np.nan}
    i = int(np.nanargmax(power))
    return {
        "period": float(np.asarray(r.period)[i]),
        "snr": float(np.asarray(r.depth_snr)[i]),
        "duration": float(np.asarray(r.duration)[i]),
        "t0": float(np.asarray(r.transit_time)[i]),
        "depth": float(np.asarray(r.depth)[i]),
    }


def period_recovered(recovered: float, injected: float, relative_tolerance: float = 0.02, allow_harmonics: bool = True) -> bool:
    if not np.isfinite(recovered) or recovered <= 0 or injected <= 0:
        return False
    ratios = [1.0]
    if allow_harmonics:
        ratios.extend([0.5, 2.0])
    for ratio in ratios:
        target = injected * ratio
        if abs(recovered - target) / target <= relative_tolerance:
            return True
    return False


def extract_period_candidates(periods: np.ndarray, power: np.ndarray, n_candidates: int = 5) -> pd.DataFrame:
    p = np.asarray(periods, dtype=float)
    s = np.asarray(power, dtype=float)
    good = np.isfinite(p) & np.isfinite(s) & (p > 0)
    p, s = p[good], s[good]
    if len(p) == 0:
        return pd.DataFrame(columns=["Rank", "Period_Days", "Power"])
    order = np.argsort(p)
    p, s = p[order], s[order]
    peaks, _ = find_peaks(s)
    if len(peaks) == 0:
        peaks = np.arange(len(s))
    ranked = peaks[np.argsort(s[peaks])[::-1]]
    chosen: List[int] = []
    for idx in ranked:
        candidate = p[idx]
        # Avoid listing nearly duplicate grid points around the same peak.
        if any(abs(candidate - p[j]) / candidate < 0.005 for j in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= n_candidates:
            break
    return pd.DataFrame({
        "Rank": np.arange(1, len(chosen) + 1),
        "Period_Days": [p[i] for i in chosen],
        "Power": [s[i] for i in chosen],
    })


def refine_and_vet_candidate(t: np.ndarray, flux: np.ndarray, err: np.ndarray, period: float, duration_guess: float) -> Dict[str, Any]:
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    cadence = max(float(np.median(np.diff(np.sort(t)))), 1e-5)
    duration_guess = max(float(duration_guess), 2.0 * cadence)
    durations = np.unique(np.clip(duration_guess * np.array([0.5, 0.7, 1.0, 1.3, 1.6]), 1.5 * cadence, 0.25 * period))
    durations = durations[(durations > cadence) & (durations < period)]
    bls = BoxLeastSquares(t, flux, dy=np.maximum(err, 1e-12))
    pgrid = np.linspace(period * 0.995, period * 1.005, 101)
    res = bls.power(pgrid, durations, objective="snr", oversample=10)
    i = int(np.nanargmax(np.asarray(res.power, dtype=float)))
    p = float(np.asarray(res.period)[i])
    dur = float(np.asarray(res.duration)[i])
    tc = float(np.asarray(res.transit_time)[i])
    depth = float(np.asarray(res.depth)[i])
    depth_err = float(np.asarray(res.depth_err)[i])
    depth_snr = float(np.asarray(res.depth_snr)[i])
    stats = bls.compute_stats(p, dur, tc)

    def pair(name: str) -> Tuple[float, float]:
        value = stats.get(name, (np.nan, np.nan))
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return np.nan, np.nan

    odd, odd_e = pair("depth_odd")
    even, even_e = pair("depth_even")
    secondary, secondary_e = pair("depth_phased")
    half, half_e = pair("depth_half")
    denom = math.sqrt(max(odd_e, 0) ** 2 + max(even_e, 0) ** 2)
    odd_even_sigma = abs(odd - even) / denom if denom > 0 else np.nan
    secondary_snr = secondary / secondary_e if secondary_e > 0 else np.nan
    ntrans = _count_transits_in_baseline(t, tc, p)
    return {
        "period": p, "duration": dur, "t0": tc, "depth": depth, "depth_err": depth_err,
        "depth_snr": depth_snr, "odd_depth": odd, "even_depth": even,
        "odd_even_sigma": odd_even_sigma, "secondary_depth": secondary,
        "secondary_snr": secondary_snr, "half_period_depth": half, "n_transits": ntrans,
    }


def estimate_depth_at_ephemeris(t: np.ndarray, flux: np.ndarray, err: np.ndarray, t0: float, period: float, duration: float) -> Dict[str, float]:
    phase = np.abs(phase_from_ephemeris(t, t0, period))
    in_m = phase <= 0.5 * duration
    out_m = (phase >= 1.5 * duration) & (phase <= min(0.4 * period, 5.0 * duration))
    if np.sum(in_m) < 2 or np.sum(out_m) < 3:
        return {"depth": np.nan, "error": np.nan, "snr": np.nan, "n_in": int(np.sum(in_m))}
    wi = 1.0 / np.maximum(err[in_m], 1e-12) ** 2
    wo = 1.0 / np.maximum(err[out_m], 1e-12) ** 2
    fi = np.sum(wi * flux[in_m]) / np.sum(wi)
    fo = np.sum(wo * flux[out_m]) / np.sum(wo)
    ei = math.sqrt(1.0 / np.sum(wi))
    eo = math.sqrt(1.0 / np.sum(wo))
    depth = fo - fi
    de = math.sqrt(ei**2 + eo**2)
    return {"depth": float(depth), "error": float(de), "snr": float(depth / de) if de > 0 else np.nan, "n_in": int(np.sum(in_m))}


def q_to_u(q1: float, q2: float) -> Tuple[float, float]:
    sq = math.sqrt(max(q1, 0.0))
    return 2.0 * sq * q2, sq * (1.0 - 2.0 * q2)


def u_to_q(u1: float, u2: float) -> Tuple[float, float]:
    s = u1 + u2
    if s <= 0:
        return 0.25, 0.5
    q1 = s**2
    q2 = u1 / (2.0 * s)
    return float(np.clip(q1, 1e-6, 1 - 1e-6)), float(np.clip(q2, 1e-6, 1 - 1e-6))


@dataclass(frozen=True)
class AdvancedMCMCBounds:
    rp_min: float
    rp_max: float
    b_min: float
    b_max: float
    t0_min: float
    t0_max: float
    p_min: float
    p_max: float
    log_jitter_min: float = math.log(1e-8)
    log_jitter_max: float = math.log(0.1)
    offset_min: float = -0.05
    offset_max: float = 0.05


def advanced_log_posterior(
    theta: np.ndarray, t: np.ndarray, flux: np.ndarray, err: np.ndarray,
    engine: TransitEngine, bounds: AdvancedMCMCBounds, fit_ld: bool,
    ld_prior: Optional[Tuple[float, float, float, float]] = None,
) -> float:
    if fit_ld:
        rp, b, t0, period, log_jit, offset, q1, q2 = theta
        if not (0 < q1 < 1 and 0 < q2 < 1):
            return -np.inf
        u1, u2 = q_to_u(q1, q2)
        trial_engine = replace(engine, u1=u1, u2=u2)
    else:
        rp, b, t0, period, log_jit, offset = theta
        trial_engine = engine

    if not (bounds.rp_min < rp < bounds.rp_max): return -np.inf
    if not (bounds.b_min <= b < min(bounds.b_max, 1.0 + rp)): return -np.inf
    if not (bounds.t0_min < t0 < bounds.t0_max): return -np.inf
    if not (bounds.p_min < period < bounds.p_max): return -np.inf
    if not (bounds.log_jitter_min < log_jit < bounds.log_jitter_max): return -np.inf
    if not (bounds.offset_min < offset < bounds.offset_max): return -np.inf

    model = trial_engine.primary_model_from_b(t, rp, b, t0, period)
    if np.any(~np.isfinite(model)):
        return -np.inf
    model = model + offset
    jitter = math.exp(log_jit)
    sigma2 = np.maximum(err, 1e-12)**2 + jitter**2
    resid = flux - model
    lp = -0.5 * np.sum(resid**2 / sigma2 + np.log(2 * np.pi * sigma2))

    # Optional Gaussian prior on u1/u2, represented as means and sigmas.
    if fit_ld and ld_prior is not None:
        mu1, sig1, mu2, sig2 = ld_prior
        if sig1 > 0: lp += -0.5 * ((u1 - mu1) / sig1)**2
        if sig2 > 0: lp += -0.5 * ((u2 - mu2) / sig2)**2
    return float(lp)


def advanced_bounds_from_result(result: Dict[str, Any], engine: TransitEngine) -> AdvancedMCMCBounds:
    p0 = float(result["period"]); t0 = float(result["T0"])
    duration = max(float(result.get("duration", 0.05 * p0)), 1e-5)
    pu = float(result.get("period_uncertainty", np.nan))
    if not np.isfinite(pu) or pu <= 0: pu = 0.002 * p0
    return AdvancedMCMCBounds(
        rp_min=0.001, rp_max=min(1.5, max(0.3, 2.5 * max(engine.rp_rs, 0.05))),
        b_min=0.0, b_max=2.0,
        t0_min=t0 - max(3 * duration, 0.03 * p0), t0_max=t0 + max(3 * duration, 0.03 * p0),
        p_min=max(1e-5, p0 - max(7 * pu, 0.03 * p0)), p_max=p0 + max(7 * pu, 0.03 * p0),
    )


def initialize_advanced_walkers(
    n_walkers: int, result: Dict[str, Any], engine: TransitEngine,
    bounds: AdvancedMCMCBounds, err: np.ndarray, fit_ld: bool, seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rp0 = float(result.get("rp_rs", np.nan))
    if not np.isfinite(rp0) or rp0 <= 0:
        depth_level = float(result.get("depth", 1 - engine.rp_rs**2))
        rp0 = math.sqrt(max(1 - depth_level, 1e-6))
    rp0 = float(np.clip(rp0, bounds.rp_min * 2, bounds.rp_max * 0.8))
    b0 = engine.impact_parameter(period_days=float(result["period"]))
    if not np.isfinite(b0) or b0 < 0 or b0 >= 1 + rp0: b0 = min(0.4, 0.5 * (1 + rp0))
    p0 = float(result["period"]); t0 = float(result["T0"])
    duration = max(float(result.get("duration", 0.05 * p0)), 1e-5)
    pu = float(result.get("period_uncertainty", np.nan))
    if not np.isfinite(pu) or pu <= 0: pu = 0.001 * p0
    logj = math.log(max(float(np.median(err)) * 0.5, 1e-7))
    center = [rp0, b0, t0, p0, logj, 0.0]
    scale = [max(0.04 * rp0, 5e-4), 0.08, max(0.15 * duration, 1e-5), max(pu, 1e-6), 0.35, 0.002]
    if fit_ld:
        q1, q2 = u_to_q(engine.u1, engine.u2)
        center += [q1, q2]; scale += [0.05, 0.05]
    center = np.asarray(center); scale = np.asarray(scale)
    ndim = len(center)
    pos = np.empty((n_walkers, ndim), dtype=float)
    for i in range(n_walkers):
        for _ in range(10000):
            trial = center + scale * rng.normal(size=ndim)
            if fit_ld:
                basic = trial[:6]
                valid_ld = 0 < trial[6] < 1 and 0 < trial[7] < 1
            else:
                basic = trial
                valid_ld = True
            rp,b,t0t,pt,lj,off = basic
            if (bounds.rp_min < rp < bounds.rp_max and bounds.b_min <= b < min(bounds.b_max,1+rp)
                and bounds.t0_min < t0t < bounds.t0_max and bounds.p_min < pt < bounds.p_max
                and bounds.log_jitter_min < lj < bounds.log_jitter_max and bounds.offset_min < off < bounds.offset_max
                and valid_ld):
                pos[i] = trial; break
        else:
            raise RuntimeError("Could not initialize advanced MCMC walkers inside priors.")
    return pos


def mcmc_information_criteria(
    t: np.ndarray, flux: np.ndarray, err: np.ndarray, engine: TransitEngine,
    median_params: Dict[str, float], fit_ld: bool,
) -> Dict[str, float]:
    rp = median_params["Rp/Rs"]; b = median_params["b"]; t0 = median_params["T0"]; p = median_params["Period"]
    jitter = math.exp(median_params["log_jitter"]); offset = median_params.get("offset", 0.0)
    trial = engine
    if fit_ld and "q1" in median_params and "q2" in median_params:
        u1,u2 = q_to_u(median_params["q1"], median_params["q2"]); trial=replace(engine,u1=u1,u2=u2)
    model = trial.primary_model_from_b(t,rp,b,t0,p)+offset
    sigma2=np.maximum(err,1e-12)**2+jitter**2
    logl_trans=-0.5*np.sum((flux-model)**2/sigma2+np.log(2*np.pi*sigma2))
    # Null model gets its own weighted constant but shares jitter for a fair quick diagnostic.
    w=1/sigma2; c=np.sum(w*flux)/np.sum(w)
    logl_null=-0.5*np.sum((flux-c)**2/sigma2+np.log(2*np.pi*sigma2))
    n=max(len(t),1); k_trans=8 if fit_ld else 6; k_null=2
    bic_trans=k_trans*np.log(n)-2*logl_trans; bic_null=k_null*np.log(n)-2*logl_null
    return {"BIC_transit":float(bic_trans),"BIC_null":float(bic_null),"Delta_BIC_null_minus_transit":float(bic_null-bic_trans)}


def utc_time_to_bjd_tdb(time_utc: Time, coord: SkyCoord, location: EarthLocation) -> np.ndarray:
    t = Time(time_utc, location=location)
    ltt = t.light_travel_time(coord, kind="barycentric")
    return np.asarray((t.tdb + ltt).jd, dtype=float)


def bjd_tdb_to_utc_time(bjd: float, coord: SkyCoord, location: EarthLocation) -> Time:
    """Numerically invert BJD_TDB = UTC.tdb + barycentric light-travel time."""
    guess = Time(float(bjd), format="jd", scale="tdb", location=location).utc
    for _ in range(5):
        trial = Time(guess.jd, format="jd", scale="utc", location=location)
        computed = float((trial.tdb + trial.light_travel_time(coord, kind="barycentric")).jd)
        delta = float(bjd) - computed
        guess = Time(guess.jd + delta, format="jd", scale="utc", location=location)
    return guess


def predicted_transit_centers(t0_bjd: float, period: float, start_bjd: float, end_bjd: float) -> np.ndarray:
    k0 = int(np.floor((start_bjd - t0_bjd) / period)) - 1
    k1 = int(np.ceil((end_bjd - t0_bjd) / period)) + 1
    tcs = t0_bjd + np.arange(k0, k1 + 1) * period
    return tcs[(tcs >= start_bjd) & (tcs <= end_bjd)]


def observing_event_metrics(
    tc_bjd: float, duration_days: float, coord: SkyCoord, location: EarthLocation,
    min_alt_deg: float, max_airmass: float, min_moon_sep_deg: float,
    baseline_before_hr: float, baseline_after_hr: float,
) -> Dict[str, Any]:
    tc_utc = bjd_tdb_to_utc_time(tc_bjd, coord, location)
    total_before = baseline_before_hr / 24.0 + 0.5 * duration_days
    total_after = baseline_after_hr / 24.0 + 0.5 * duration_days
    start = tc_utc.jd - total_before
    end = tc_utc.jd + total_after
    n = max(60, int((end-start)*24*12)+1)  # about 5 min grid
    times = Time(np.linspace(start,end,n),format="jd",scale="utc",location=location)
    frame = AltAz(obstime=times, location=location)
    alt = coord.transform_to(frame).alt.deg
    secz = coord.transform_to(frame).secz.value
    sun_alt = get_sun(times).transform_to(frame).alt.deg
    moon = get_body("moon", times, location=location)
    moon_sep = moon.separation(coord).deg
    good = (alt >= min_alt_deg) & (secz >= 1.0) & (secz <= max_airmass) & (sun_alt <= -18.0) & (moon_sep >= min_moon_sep_deg)
    event_mask = np.abs(times.jd - tc_utc.jd) <= 0.5 * duration_days
    before_mask = (times.jd < tc_utc.jd - 0.5*duration_days)
    after_mask = (times.jd > tc_utc.jd + 0.5*duration_days)
    event_cov = float(np.mean(good[event_mask])) if np.any(event_mask) else 0.0
    # Contiguous usable baseline immediately adjacent to ingress/egress.
    # This is stricter and more useful than summing disconnected good samples.
    dt_hr = float(np.median(np.diff(times.jd))*24.0) if len(times)>1 else 0.0
    pre_idx = np.where(before_mask)[0]
    post_idx = np.where(after_mask)[0]
    pre_count = 0
    for idx in pre_idx[::-1]:
        if good[idx]:
            pre_count += 1
        else:
            break
    post_count = 0
    for idx in post_idx:
        if good[idx]:
            post_count += 1
        else:
            break
    pre_hr = float(pre_count * dt_hr)
    post_hr = float(post_count * dt_hr)
    mid_idx = int(np.argmin(np.abs(times.jd-tc_utc.jd)))
    return {
        "tc_bjd":float(tc_bjd), "tc_utc":tc_utc.isot, "event_coverage":event_cov,
        "pre_baseline_hr":min(pre_hr,baseline_before_hr), "post_baseline_hr":min(post_hr,baseline_after_hr),
        "altitude_mid_deg":float(alt[mid_idx]), "airmass_mid":float(secz[mid_idx]) if np.isfinite(secz[mid_idx]) else np.nan,
        "moon_sep_mid_deg":float(moon_sep[mid_idx]), "sun_alt_mid_deg":float(sun_alt[mid_idx]),
        "score":float(100*(0.6*event_cov+0.2*min(pre_hr/max(baseline_before_hr,1e-6),1)+0.2*min(post_hr/max(baseline_after_hr,1e-6),1))),
    }


def build_observing_plan(
    t0_bjd: float, period: float, duration_days: float, coord: SkyCoord, location: EarthLocation,
    start_date: date, end_date: date, min_alt_deg: float, max_airmass: float,
    min_moon_sep_deg: float, baseline_before_hr: float, baseline_after_hr: float,
    max_events: int = 100,
) -> pd.DataFrame:
    start_utc = Time(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc), scale="utc", location=location)
    end_utc = Time(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc), scale="utc", location=location)
    start_bjd = float(utc_time_to_bjd_tdb(start_utc, coord, location))
    end_bjd = float(utc_time_to_bjd_tdb(end_utc, coord, location))
    centers = predicted_transit_centers(t0_bjd, period, start_bjd, end_bjd)[:max_events]
    rows = [observing_event_metrics(
        tc,duration_days,coord,location,min_alt_deg,max_airmass,min_moon_sep_deg,baseline_before_hr,baseline_after_hr
    ) for tc in centers]
    return pd.DataFrame(rows)


def infer_date_from_filename(filename: str) -> Optional[date]:
    name = str(filename)
    patterns = [r"(20\d{2})[._-]?(\d{2})[._-]?(\d{2})", r"(20\d{2})\.(\d{2})(\d{2})"]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            try: return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
            except ValueError: pass
    return None


def process_night_dataframe(
    df: pd.DataFrame, time_col: str, flux_col: str, err_choice: str, flag_choice: str,
    time_format: str, frame_cadence_min: float,
) -> Tuple[np.ndarray,np.ndarray,np.ndarray,Dict[str,int]]:
    raw_t=df[time_col].to_numpy(); internal_t,_,_,_=to_internal_time(raw_t,time_format,frame_cadence_min=frame_cadence_min)
    raw_flux=pd.to_numeric(df[flux_col],errors="coerce").to_numpy(dtype=float)
    raw_err=None if err_choice=="Estimate from robust scatter" else pd.to_numeric(df[err_choice],errors="coerce").to_numpy(dtype=float)
    if flag_choice!="Do not use a quality flag":
        bad=boolean_bad_row_mask(df[flag_choice]); keep=~bad
        internal_t,raw_flux=internal_t[keep],raw_flux[keep]
        if raw_err is not None: raw_err=raw_err[keep]
    return prepare_lightcurve(internal_t,raw_flux,raw_err,clip_positive_spikes=False)


def dataframe_to_fits_bytes(df: pd.DataFrame) -> bytes:
    table = Table.from_pandas(df)
    bio = io.BytesIO()
    table.write(bio, format="fits")
    return bio.getvalue()


def matplotlib_figure_bytes(fig: plt.Figure, fmt: str = "png") -> bytes:
    bio=io.BytesIO(); fig.savefig(bio,format=fmt,bbox_inches="tight",dpi=180); bio.seek(0); return bio.getvalue()


def analysis_methods_text(
    engine: TransitEngine, observatory_name: str, search_result: Dict[str,Any], gp_kernel: str,
    detection_mode: str, time_label: str, n_points: int,
) -> str:
    return f"""VORTEX Professional {APP_VERSION} analysis summary\n\nData:\n- {n_points} usable photometric measurements\n- Internal time unit: days; displayed time convention: {time_label}\n\nForward model:\n- BATMAN analytic transit model\n- Host mass: {engine.host_mass_jup:.6g} M_Jup\n- Host radius: {engine.host_radius_jup:.6g} R_Jup\n- Radius ratio: {engine.rp_rs:.7f}\n- Eccentricity: {engine.eccentricity:.6g}\n- Argument of periastron: {engine.omega_deg:.6g} deg\n- Quadratic limb darkening: u1={engine.u1:.6g}, u2={engine.u2:.6g}\n- Finite exposure integration: {engine.exposure_time_days*86400:.3f} s, supersample={engine.supersample_factor}\n\nObservatory simulation profile:\n- {observatory_name}\n- Profile noise values are illustrative unless explicitly replaced by measured/instrument-specific values.\n\nDetrending:\n- Transit-aware Gaussian process trained outside candidate transit windows\n- Per-point variances supplied through GaussianProcessRegressor alpha\n- Fitted kernel: {gp_kernel}\n\nPeriod search:\n- Detection mode: {detection_mode}\n- Search method: {search_result.get('search_method','unknown')}\n- Period: {float(search_result.get('period',np.nan)):.9g} d\n- T0: {float(search_result.get('T0',np.nan)):.9g}\n- Duration: {float(search_result.get('duration',np.nan)):.9g} d\n\nCaveats:\n- Broad-band thermal emission uses blackbody integration; a user throughput curve is used when supplied.\n- The quick ETC is approximate and is not a replacement for official instrument ETCs.\n- Lazuli continuous coverage is an idealized comparison benchmark, not an operational pointing/visibility model.\n"""


def build_analysis_zip(
    config: Dict[str,Any], reduced_df: pd.DataFrame, periodogram_df: pd.DataFrame,
    candidates_df: pd.DataFrame, methods_text: str, posterior_df: Optional[pd.DataFrame]=None,
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> bytes:
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("vortex_project.json",json.dumps(config,indent=2,default=str))
        z.writestr("reduced_lightcurve.csv",reduced_df.to_csv(index=False))
        z.writestr("periodogram.csv",periodogram_df.to_csv(index=False))
        z.writestr("candidates.csv",candidates_df.to_csv(index=False))
        z.writestr("methods.txt",methods_text)
        if posterior_df is not None:
            z.writestr("posterior_samples.csv",posterior_df.to_csv(index=False))
        if extra_tables:
            for name, table in extra_tables.items():
                if isinstance(table, pd.DataFrame):
                    z.writestr(str(name), table.to_csv(index=False))
    return bio.getvalue()


def apply_project_state(config: Dict[str,Any]) -> None:
    """Queue widget values before they are instantiated on the next rerun."""
    st.session_state["pending_project_config"] = config


def consume_pending_project_state() -> None:
    cfg=st.session_state.pop("pending_project_config",None)
    if not isinstance(cfg,dict): return
    for key,value in cfg.get("widget_state",{}).items():
        st.session_state[key]=value


def search_lightkurve_products(target: str, mission: str) -> Any:
    if not LIGHTKURVE_AVAILABLE:
        raise RuntimeError("Lightkurve is not installed. Install package 'lightkurve'.")
    return lk.search_lightcurve(target, mission=mission, limit=50)


def download_lightkurve_product(search_result: Any, index: int) -> pd.DataFrame:
    lc = search_result[int(index)].download(quality_bitmask="default")
    if lc is None:
        raise RuntimeError("Lightkurve returned no light curve for the selected product.")
    lc = lc.remove_nans().normalize()
    t = np.asarray(lc.time.tdb.jd, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    if getattr(lc,"flux_err",None) is not None:
        e = np.asarray(lc.flux_err.value,dtype=float)
    else:
        e = np.full_like(f,np.nan)
    return pd.DataFrame({"BJD_TDB":t,"NormalizedFlux":f,"FluxError":e})


def injection_recovery_grid(
    t: np.ndarray, base_flux: np.ndarray, err: np.ndarray, base_engine: TransitEngine,
    periods: np.ndarray, radii_earth: np.ndarray, trials_per_cell: int,
    snr_threshold: float, tolerance: float, allow_harmonics: bool, seed: int,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    rng=np.random.default_rng(seed); rows=[]; detail=[]
    span=float(np.ptp(t)); cadence=max(float(np.median(np.diff(np.sort(t)))),1e-5)
    for radius in radii_earth:
        for period in periods:
            recovered=0; snrs=[]
            if span < 2*period:
                rows.append({"Radius_REarth":radius,"Period_Days":period,"Completeness":0.0,"Median_SNR":np.nan,"N":trials_per_cell,"Reason":"<2 periods in baseline"})
                continue
            trial_engine=replace(base_engine,companion_radius_earth=float(radius),period_days=float(period))
            duration=trial_engine.transit_duration_days(period_days=float(period))
            if duration<=0:
                rows.append({"Radius_REarth":radius,"Period_Days":period,"Completeness":0.0,"Median_SNR":np.nan,"N":trials_per_cell,"Reason":"non-transiting geometry"})
                continue
            for j in range(trials_per_cell):
                t0=float(t.min()+rng.uniform(0,period))
                model=trial_engine.generate_light_curve(t,t0=t0,period_days=float(period),include_thermal_phase=False)
                injected=base_flux*model
                # Fast search; bound to periods with >=2 events.
                pmin=max(3*cadence,0.5*period); pmax=min(2*period,span/2)
                r=fast_bls_candidate(t,injected,err,pmin,pmax,duration)
                ok=period_recovered(r["period"],period,tolerance,allow_harmonics) and np.isfinite(r["snr"]) and r["snr"]>=snr_threshold
                recovered+=int(ok); snrs.append(r["snr"])
                detail.append({"Radius_REarth":radius,"Period_Days":period,"Trial":j,"Injected_T0":t0,"Recovered_Period":r["period"],"Recovered_SNR":r["snr"],"Recovered":ok})
            rows.append({"Radius_REarth":radius,"Period_Days":period,"Completeness":recovered/trials_per_cell,"Median_SNR":float(np.nanmedian(snrs)) if len(snrs) else np.nan,"N":trials_per_cell,"Reason":""})
    return pd.DataFrame(rows),pd.DataFrame(detail)

# =============================================================================
# 14. STREAMLIT PROFESSIONAL UI
# =============================================================================

consume_pending_project_state()

st.set_page_config(
    page_title="VORTEX Professional",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Global app */
      .stApp {
        background-color: #05080e;
        color: #dbe8f5;
      }

      html, body, [class*="css"] {
        color: #dbe8f5;
      }

      /* Main text */
      h1, h2, h3, h4, h5, h6,
      p, span, div, label, li, small {
        color: #dbe8f5;
      }

      h1, h2, h3, h4 {
        letter-spacing: .02em;
      }

      /* Captions / secondary text */
      .stCaption, .vortex-small {
        color: #8fa8c6 !important;
        opacity: 1 !important;
        font-size: .88rem;
      }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #090e18;
      }

      section[data-testid="stSidebar"] * {
        color: #dbe8f5 !important;
      }

      /* Metric cards */
      div[data-testid="stMetric"] {
        background: #0b111e;
        border: 1px solid #1a273e;
        border-radius: 8px;
        padding: 10px 14px;
      }

      div[data-testid="stMetricValue"] {
        color: #79e6f2 !important;
      }

      div[data-testid="stMetricLabel"] {
        color: #8fa8c6 !important;
      }

      div[data-testid="stMetricLabel"] * {
        color: #8fa8c6 !important;
      }

      /* Markdown containers */
      [data-testid="stMarkdownContainer"] * {
        color: #dbe8f5 !important;
      }

      /* Tabs */
      button[data-baseweb="tab"] {
        color: #8fa8c6 !important;
      }

      button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f0ff !important;
        border-bottom: 2px solid #00f0ff !important;
      }

      /* Expander */
      details, summary {
        color: #dbe8f5 !important;
      }

      /* Inputs and selectboxes */
      .stSelectbox label,
      .stNumberInput label,
      .stSlider label,
      .stCheckbox label,
      .stRadio label,
      .stTextInput label,
      .stTextArea label {
        color: #dbe8f5 !important;
      }

      [data-baseweb="select"] * {
        color: #dbe8f5 !important;
      }

      input, textarea {
        color: #dbe8f5 !important;
        background-color: #0b111e !important;
      }

      /* Buttons */
      .stButton > button {
        background: #00f0ff;
        color: #041018 !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
      }

      .stButton > button:hover {
        background: #00d2e0;
        color: #041018 !important;
      }

      /* Custom info boxes */
      .vortex-note {
        padding: .65rem .8rem;
        border: 1px solid #26364e;
        border-radius: 7px;
        background: #0b111e;
        color: #dbe8f5 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔭 VORTEX Professional")
st.caption(
    "Visual Observatory for Retrieval & Transit Exploration — simulation, observing strategy, "
    "real-data analysis, injection/recovery, Bayesian inference, and reproducible export."
)

# Session objects that should survive ordinary Streamlit reruns.
for _key, _default in {
    "archive_results": None,
    "mission_search": None,
    "mission_df": None,
    "mcmc_pro": None,
    "mcmc_meta": None,
    "compare_result": None,
    "planner_result": None,
    "injrec_result": None,
    "injrec_detail": None,
    "multi_result": None,
    "multi_mcmc": None,
    "manual_excluded_rows": [],
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def _set_target_state(values: Dict[str, Any]) -> None:
    mapping = {
        "host_mass_jup": "host_mass_jup",
        "host_radius_jup": "host_radius_jup",
        "companion_radius_earth": "companion_radius_earth",
        "companion_mass_earth": "companion_mass_earth",
        "period_days": "period_days",
        "inclination_deg": "inclination_deg",
        "eccentricity": "eccentricity",
        "omega_deg": "omega_deg",
        "host_temp_k": "host_temp_k",
        "companion_temp_k": "companion_temp_k",
        "logg": "target_logg",
        "metallicity": "target_metallicity",
        "ra_deg": "target_ra_deg",
        "dec_deg": "target_dec_deg",
        "archive_t0_bjd": "archive_t0_bjd",
    }
    for src, dst in mapping.items():
        if src in values and values[src] is not None:
            value = values[src]
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            st.session_state[dst] = value


# -----------------------------------------------------------------------------
# SIDEBAR: workflow / target / observatory / model / data
# -----------------------------------------------------------------------------

st.sidebar.header("VORTEX WORKFLOW")
workflow = st.sidebar.radio(
    "Mode",
    ["Explore / Simulate", "Analyze Real Data", "Inject & Recover"],
    key="workflow_mode",
    help="The simulator remains available in Explore mode; professional data workflows add analysis layers without replacing it.",
)

st.sidebar.divider()
st.sidebar.header("🎯 TARGET SYSTEM")
target_source = st.sidebar.selectbox(
    "Parameter source",
    ["Built-in preset", "NASA Exoplanet Archive", "Custom"],
    key="target_source",
)

default_preset_name = "VHS 1256 b + Hypothetical Transiting Satellite"
if target_source == "Built-in preset":
    preset_name = st.sidebar.selectbox(
        "Preset",
        list(TARGET_PRESETS.keys()),
        index=list(TARGET_PRESETS.keys()).index(default_preset_name),
        key="target_preset_name",
    )
    if st.sidebar.button("Load preset parameters", key="load_preset_btn", use_container_width=True):
        _set_target_state(TARGET_PRESETS[preset_name])
        st.session_state["target_display_name"] = preset_name
        st.rerun()
elif target_source == "NASA Exoplanet Archive":
    archive_query = st.sidebar.text_input("Planet or host name", value="TRAPPIST-1 c", key="archive_query")
    if st.sidebar.button("Search NASA Archive", key="archive_search_btn", use_container_width=True):
        try:
            st.session_state.archive_results = query_nasa_exoplanet_archive(archive_query)
        except Exception as exc:
            st.sidebar.error(str(exc))
    if isinstance(st.session_state.archive_results, pd.DataFrame) and not st.session_state.archive_results.empty:
        archive_df = st.session_state.archive_results
        options = [f"{i}: {row['pl_name']}" for i, row in archive_df.iterrows()]
        choice = st.sidebar.selectbox("Archive match", options, key="archive_match_choice")
        row_idx = int(choice.split(":", 1)[0])
        if st.sidebar.button("Load selected archive values", key="archive_load_btn", use_container_width=True):
            row = archive_df.loc[row_idx]
            _set_target_state(archive_row_to_target(row))
            st.session_state["target_display_name"] = str(row.get("pl_name", archive_query))
            st.rerun()
else:
    st.sidebar.caption("Custom parameters are entered below.")

# Initialize parameter state once; do not overwrite later user edits.
_default_target = TARGET_PRESETS[default_preset_name]
for _src, _key in {
    "host_mass_jup": "host_mass_jup", "host_radius_jup": "host_radius_jup",
    "companion_radius_earth": "companion_radius_earth", "companion_mass_earth": "companion_mass_earth",
    "period_days": "period_days", "inclination_deg": "inclination_deg", "eccentricity": "eccentricity",
    "omega_deg": "omega_deg", "host_temp_k": "host_temp_k", "companion_temp_k": "companion_temp_k",
    "logg": "target_logg", "metallicity": "target_metallicity", "ra_deg": "target_ra_deg", "dec_deg": "target_dec_deg",
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default_target[_src]
if "target_display_name" not in st.session_state:
    st.session_state.target_display_name = default_preset_name
if "archive_t0_bjd" not in st.session_state:
    st.session_state.archive_t0_bjd = np.nan

with st.sidebar.expander("Physical parameters", expanded=True):
    host_mass_jup = st.number_input("Host mass (M_Jup)", min_value=0.001, step=0.1, format="%.6f", key="host_mass_jup")
    host_radius_jup = st.number_input("Host radius (R_Jup)", min_value=0.01, step=0.01, format="%.6f", key="host_radius_jup")
    companion_radius_earth = st.number_input("Companion radius (R_Earth)", min_value=0.01, step=0.05, format="%.5f", key="companion_radius_earth")
    companion_mass_earth = st.number_input("Companion mass (M_Earth; optional for Kepler's law)", min_value=0.0, step=0.1, format="%.5f", key="companion_mass_earth")
    period_days = st.number_input("Orbital period (days)", min_value=1e-4, step=0.01, format="%.9f", key="period_days")
    inclination_deg = st.slider("Inclination (deg)", 60.0, 90.0, step=0.01, key="inclination_deg")
    eccentricity = st.slider("Eccentricity", 0.0, 0.95, step=0.01, key="eccentricity")
    omega_deg = st.slider("Argument of periastron ω (deg)", 0.0, 360.0, step=1.0, key="omega_deg")

with st.sidebar.expander("Target atmosphere / coordinates", expanded=False):
    host_temp_k = st.number_input("Host effective temperature (K)", min_value=100.0, max_value=50000.0, step=50.0, key="host_temp_k")
    companion_temp_k = st.number_input("Companion brightness-temperature proxy (K)", min_value=10.0, max_value=10000.0, step=25.0, key="companion_temp_k")
    target_logg = st.number_input("Host log(g), cgs", min_value=0.0, max_value=6.5, step=0.05, key="target_logg")
    target_metallicity = st.number_input("Host [M/H]", min_value=-3.0, max_value=1.5, step=0.05, key="target_metallicity")
    target_ra_deg = st.number_input("Target RA (deg)", min_value=0.0, max_value=360.0, step=0.001, format="%.7f", key="target_ra_deg")
    target_dec_deg = st.number_input("Target Dec (deg)", min_value=-90.0, max_value=90.0, step=0.001, format="%.7f", key="target_dec_deg")

st.sidebar.divider()
st.sidebar.header("📡 OBSERVATORY / BANDPASS")
obs_name = st.sidebar.selectbox("Telescope / instrument profile", ObservatoryProfiles.names(), key="observatory_name")
obs_p = ObservatoryProfiles.get_profile(obs_name)

# Custom observatory overrides are persistent widget values.
if obs_name.startswith("Custom"):
    with st.sidebar.expander("Custom observatory definition", expanded=True):
        custom_diameter = st.number_input("Aperture diameter (m)", min_value=0.1, value=float(obs_p["diameter_m"]), step=0.1, key="custom_diameter")
        custom_cadence = st.number_input("Default cadence (min)", min_value=0.001, value=float(obs_p["cadence_min"]), step=0.1, key="custom_cadence")
        custom_hours = st.number_input("Daily observing window (h)", min_value=0.1, max_value=24.0, value=float(obs_p["night_hours"]), step=0.5, key="custom_hours")
        obs_p.update({"diameter_m": custom_diameter, "cadence_min": custom_cadence, "night_hours": custom_hours})
        if not obs_p["space"]:
            obs_p["lat_deg"] = st.number_input("Latitude (deg)", -90.0, 90.0, value=float(obs_p["lat_deg"]), format="%.6f", key="custom_lat")
            obs_p["lon_deg"] = st.number_input("Longitude (deg; east +)", -180.0, 180.0, value=float(obs_p["lon_deg"]), format="%.6f", key="custom_lon")
            obs_p["height_m"] = st.number_input("Elevation (m)", -500.0, 10000.0, value=float(obs_p["height_m"]), key="custom_height")

bandpass = st.sidebar.selectbox("Bandpass", list(BANDS.keys()), key="bandpass")
throughput_mode = st.sidebar.selectbox("Thermal band model", ["Top-hat approximation", "Upload throughput curve"], key="throughput_mode")
throughput_wav = None
throughput_val = None
throughput_df = None
if throughput_mode == "Upload throughput curve":
    throughput_file = st.sidebar.file_uploader("Throughput CSV (wavelength, throughput)", type=["csv", "txt"], key="throughput_file")
    if throughput_file is not None:
        try:
            throughput_wav, throughput_val, throughput_df = parse_throughput_upload(throughput_file)
            st.sidebar.success(f"Loaded {len(throughput_wav)} throughput samples.")
        except Exception as exc:
            st.sidebar.error(f"Throughput file: {exc}")

band_cfg = BANDS[bandpass]
selected_max_um = band_cfg["center_um"] + 0.5 * band_cfg["width_um"]
if "Lazuli" in obs_name and selected_max_um > float(obs_p.get("band_max_um", np.inf)):
    st.sidebar.warning(
        "This band extends beyond the public ~0.4–1.7 μm Lazuli baseline. "
        "VORTEX will calculate it only as a hypothetical/customized band, not as a published Lazuli capability."
    )

with st.sidebar.expander("Limb darkening", expanded=False):
    ld_mode = st.selectbox("Quadratic limb-darkening source", ["Approximate band defaults", "Custom u1/u2", "ExoTiC-LD"], key="ld_mode")
    if ld_mode == "Approximate band defaults":
        u1, u2 = APPROX_LD[bandpass]
        st.caption(f"Approximate defaults: u1={u1:.3f}, u2={u2:.3f}")
    elif ld_mode == "Custom u1/u2":
        u1 = st.number_input("u1", value=float(APPROX_LD[bandpass][0]), step=0.01, key="custom_u1")
        u2 = st.number_input("u2", value=float(APPROX_LD[bandpass][1]), step=0.01, key="custom_u2")
    else:
        exotic_mode = st.selectbox(
            "ExoTiC-LD instrument mode",
            ["JWST_NIRSpec_Prism", "TESS", "custom"],
            key="exotic_mode",
            help="Choose custom when supplying your own wavelength-throughput curve.",
        )
        if "exotic_u1" not in st.session_state:
            st.session_state.exotic_u1, st.session_state.exotic_u2 = APPROX_LD[bandpass]
        if st.button("Calculate ExoTiC-LD coefficients", key="calc_exotic_ld"):
            try:
                if exotic_mode == "custom" and (throughput_wav is None or throughput_val is None):
                    raise ValueError("Custom ExoTiC-LD mode requires an uploaded throughput curve.")
                wl_min = band_cfg["center_um"] - 0.5 * band_cfg["width_um"]
                wl_max = band_cfg["center_um"] + 0.5 * band_cfg["width_um"]
                custom_w_ang = tuple(float(x) * 1e4 for x in throughput_wav) if throughput_wav is not None else None
                ld = compute_exotic_ld_coefficients(
                    float(target_metallicity), float(host_temp_k), float(target_logg),
                    "mps1", exotic_mode, float(wl_min) * 1e4, float(wl_max) * 1e4,
                    custom_w_ang, throughput_val,
                )
                st.session_state.exotic_u1, st.session_state.exotic_u2 = ld
            except Exception as exc:
                st.error(str(exc))
        u1, u2 = float(st.session_state.exotic_u1), float(st.session_state.exotic_u2)
        st.caption(f"Current coefficients: u1={u1:.5f}, u2={u2:.5f}")

if not quadratic_ld_is_physical(float(u1), float(u2)):
    st.sidebar.error("The selected quadratic limb-darkening coefficients are outside the usual physical domain.")

with st.sidebar.expander("Finite exposure / phase model", expanded=False):
    exposure_seconds = st.number_input("Exposure integration time (s)", min_value=0.0, value=60.0, step=10.0, key="model_exposure_s")
    supersample = st.select_slider("BATMAN supersampling", options=[1, 3, 5, 7, 9, 15], value=5, key="supersample")
    nightside_fraction = st.slider("Thermal nightside / dayside flux fraction", 0.0, 1.0, value=0.0, step=0.05, key="nightside_fraction")
    include_thermal_phase = st.checkbox("Include blackbody thermal phase + secondary eclipse", value=True, key="include_thermal")
    ttv_amp_minutes = st.number_input("Injected TTV amplitude (min)", min_value=0.0, value=0.0, step=0.5, key="ttv_amp")
    ttv_period_days = st.number_input("Injected TTV super-period (days)", min_value=0.01, value=10.0, step=0.5, key="ttv_period")

engine = TransitEngine(
    host_mass_jup=float(host_mass_jup),
    host_radius_jup=float(host_radius_jup),
    companion_radius_earth=float(companion_radius_earth),
    companion_mass_earth=float(companion_mass_earth),
    period_days=float(period_days),
    inclination_deg=float(inclination_deg),
    eccentricity=float(eccentricity),
    omega_deg=float(omega_deg),
    host_temp_k=float(host_temp_k),
    companion_temp_k=float(companion_temp_k),
    bandpass=bandpass,
    u1=float(u1), u2=float(u2),
    exposure_time_days=float(exposure_seconds) / DAY_S,
    supersample_factor=int(supersample),
    nightside_fraction=float(nightside_fraction),
    throughput_wavelength_um=throughput_wav,
    throughput_values=throughput_val,
)

# Physical consistency checks are warnings/errors, not hidden assumptions.
for severity, message in physical_system_checks(engine):
    if severity == "error":
        st.sidebar.error(message)
    elif severity == "warning":
        st.sidebar.warning(message)
    else:
        st.sidebar.info(message)

st.sidebar.caption(obs_p.get("coverage_note", ""))

# -----------------------------------------------------------------------------
# DATA INGESTION / SYNTHETIC OBSERVATION
# -----------------------------------------------------------------------------

uploaded_df = None
source_df_raw = None
quality_report: Dict[str, int] = {}
display_offset = 0.0
display_time_label = "Elapsed Days"
is_absolute_time = False
flagged_rows_removed = 0
frame_cadence_min = float(obs_p["cadence_min"])
source_label = "Synthetic"

if workflow == "Explore / Simulate":
    st.sidebar.divider()
    st.sidebar.header("🧪 SYNTHETIC OBSERVATION")
    sim_baseline = st.sidebar.number_input("Baseline (days)", min_value=0.05, max_value=365.0, value=3.0, step=0.5, key="sim_baseline")
    sim_cadence = st.sidebar.number_input("Cadence (minutes)", min_value=0.01, value=float(obs_p["cadence_min"]), step=0.1, key="sim_cadence")
    sim_white_ppm = st.sidebar.number_input("White-noise σ (ppm; assumption)", min_value=0.0, value=float(obs_p["white_noise_ppm"]), step=50.0, key="sim_white_ppm")
    sim_red_ppm = st.sidebar.number_input("Correlated-noise σ (ppm; assumption)", min_value=0.0, value=float(obs_p["red_noise_ppm"]), step=50.0, key="sim_red_ppm")
    sim_red_tau_hr = st.sidebar.number_input("Red-noise timescale (h)", min_value=0.01, value=2.0, step=0.25, key="sim_red_tau")
    variability_ppm = st.sidebar.number_input("Astrophysical variability semi-amplitude (ppm)", min_value=0.0, value=15000.0 if "VHS" in st.session_state.target_display_name else 1000.0, step=250.0, key="sim_var_ppm")
    variability_period_hr = st.sidebar.number_input("Variability period (h)", min_value=0.05, value=8.4, step=0.1, key="sim_var_period")

    if not bool(obs_p.get("space", False)):
        st.sidebar.markdown("**Ground observing window**")
        apply_daytime_gaps = st.sidebar.checkbox(
            "Apply daytime gaps",
            value=True,
            key="apply_daytime_gaps",
            help="When enabled, VORTEX removes daytime samples from every 24-hour cycle. "
                 "The gaps therefore affect TLS/BLS recovery, phase coverage, GP detrending, "
                 "and all downstream analysis—not just the figure.",
        )
        sim_night_hours = st.sidebar.number_input(
            "Usable observing hours per night",
            min_value=0.5,
            max_value=24.0,
            value=float(obs_p.get("night_hours", 9.0)),
            step=0.5,
            key="sim_night_hours",
            help="Idealized usable ground-based observing duration in each 24-hour cycle.",
        )
        window_phase = st.sidebar.slider(
            "Night-window phase offset (days)",
            0.0,
            1.0,
            value=0.0,
            step=0.01,
            key="window_phase",
            help="Shifts the repeating night/day window relative to the synthetic transit ephemeris.",
        )
    else:
        apply_daytime_gaps = False
        sim_night_hours = 24.0
        window_phase = 0.0

    sim_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1, key="sim_seed")

    t_obs = build_sampling_times(
        obs_p,
        float(sim_baseline),
        float(sim_cadence),
        float(window_phase),
        apply_ground_daytime_gaps=bool(apply_daytime_gaps),
        night_hours_override=float(sim_night_hours),
    )
    reference_t0_internal = 0.25 * float(period_days)
    true_model = engine.generate_light_curve(
        t_obs, t0=reference_t0_internal,
        ttv_amp_minutes=float(ttv_amp_minutes), ttv_period_days=float(ttv_period_days),
        include_thermal_phase=bool(include_thermal_phase),
    )
    red = simulate_red_noise(t_obs, float(sim_red_ppm) * 1e-6, float(sim_red_tau_hr) / 24.0, seed=int(sim_seed) + 11)
    variability = simulate_astrophysical_variability(
        t_obs, amplitude=float(variability_ppm) * 1e-6,
        rotation_period_days=float(variability_period_hr) / 24.0,
    )
    rng = np.random.default_rng(int(sim_seed))
    white = rng.normal(0.0, float(sim_white_ppm) * 1e-6, len(t_obs))
    raw_flux = true_model * (1.0 + variability + red) + white
    err_array = np.full_like(t_obs, max(float(sim_white_ppm) * 1e-6, 1e-8), dtype=float)
    quality_report = {"rows_input": len(t_obs), "rows_retained": len(t_obs)}
    source_label = f"Synthetic — {obs_name}"
    inject_signal = False

else:
    st.sidebar.divider()
    st.sidebar.header("📂 OBSERVATIONAL DATA")
    data_source = st.sidebar.radio("Data source", ["Upload CSV", "TESS / Kepler / K2 (Lightkurve)"], key="data_source")
    inject_signal = workflow == "Inject & Recover"

    if data_source == "TESS / Kepler / K2 (Lightkurve)":
        if not LIGHTKURVE_AVAILABLE:
            st.sidebar.error("Lightkurve is not installed. Install the optional professional dependencies from the supplied requirements file.")
            st.stop()
        mission_target = st.sidebar.text_input("Mission target", value="TRAPPIST-1", key="mission_target")
        mission = st.sidebar.selectbox("Mission", ["TESS", "Kepler", "K2"], key="mission_name")
        if st.sidebar.button("Search mission light curves", key="mission_search_btn", use_container_width=True):
            try:
                st.session_state.mission_search = search_lightkurve_products(mission_target, mission)
            except Exception as exc:
                st.sidebar.error(str(exc))
        sr = st.session_state.mission_search
        if sr is not None and len(sr) > 0:
            idx = st.sidebar.number_input("Search-result index", min_value=0, max_value=max(len(sr) - 1, 0), value=0, step=1, key="mission_index")
            st.sidebar.caption(str(sr[int(idx)]))
            if st.sidebar.button("Download selected mission light curve", key="mission_download_btn", use_container_width=True):
                try:
                    st.session_state.mission_df = download_lightkurve_product(sr, int(idx))
                except Exception as exc:
                    st.sidebar.error(str(exc))
        if not isinstance(st.session_state.mission_df, pd.DataFrame):
            st.info("Search and download a mission light curve from the sidebar to continue.")
            st.stop()
        source_df_raw = st.session_state.mission_df.copy()
        t_abs = source_df_raw["BJD_TDB"].to_numpy(dtype=float)
        display_offset = float(np.floor(np.nanmin(t_abs)))
        t_internal = t_abs - display_offset
        t_obs, raw_flux, err_array, quality_report = prepare_lightcurve(
            t_internal,
            source_df_raw["NormalizedFlux"].to_numpy(dtype=float),
            source_df_raw["FluxError"].to_numpy(dtype=float) if np.any(np.isfinite(source_df_raw["FluxError"])) else None,
            clip_positive_spikes=False,
        )
        display_time_label = "BJD_TDB"
        is_absolute_time = True
        source_label = f"{mission} via Lightkurve"
        frame_cadence_min = float(np.median(np.diff(np.sort(t_obs))) * 1440.0)

    else:
        uploaded_file = st.sidebar.file_uploader("Upload reduced light curve (CSV)", type=["csv"], key="science_csv")
        if uploaded_file is None:
            st.info("Upload a CSV from the sidebar to continue. VORTEX will let you explicitly map time, flux, and uncertainty columns.")
            st.stop()
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
        source_df_raw = uploaded_df.copy()
        cols = uploaded_df.columns.tolist()
        if len(cols) < 2:
            st.error("The CSV needs at least a time column and a flux column.")
            st.stop()

        st.sidebar.subheader("📊 Map CSV Columns")
        time_col = st.sidebar.selectbox("Time Array", cols, index=0, key="csv_time_col")
        flux_col = st.sidebar.selectbox("Flux Array", cols, index=default_flux_column_index(cols), key="csv_flux_col")
        err_choices = ["Estimate from robust scatter"] + cols
        err_choice = st.sidebar.selectbox("Error Array", err_choices, index=0, key="csv_err_col")

        flag_choices = ["Do not use a quality flag"] + cols
        preferred_flag = next(
            (c for c in cols if str(c).strip().lower().replace(" ", "_") in {"outlierremoved", "outlier_removed", "bad_row", "quality_flag"}),
            None,
        )
        flag_default = flag_choices.index(preferred_flag) if preferred_flag in flag_choices else 0
        flag_choice = st.sidebar.selectbox("Quality / outlier flag", flag_choices, index=flag_default, key="csv_flag_col")

        suggested_time_format = guess_time_format_from_column(time_col, uploaded_df[time_col].to_numpy())
        time_format = suggested_time_format
        st.sidebar.caption(f"Detected time interpretation: **{suggested_time_format}**")
        frame_cadence_min = float(obs_p["cadence_min"])
        if time_format == "Frame Number":
            frame_cadence_min = st.sidebar.number_input(
                "Frame cadence (minutes)", min_value=0.001, value=float(obs_p["cadence_min"]), step=0.1, key="frame_cadence_min"
            )

        convert_to_bjd = False
        time_coord = None
        time_location = None
        with st.sidebar.expander("Advanced time handling", expanded=False):
            override = st.checkbox("Override automatic time interpretation", value=False, key="override_time")
            time_options = ["Frame Number", "Elapsed Minutes", "Elapsed Hours", "Elapsed Days", "BJD_TDB", "JD_UTC", "MJD_UTC"]
            if override:
                time_format = st.selectbox("Interpret selected time as", time_options, index=time_options.index(suggested_time_format), key="time_override_format")
                if time_format == "Frame Number":
                    frame_cadence_min = st.number_input("Frame cadence override (min)", min_value=0.001, value=float(frame_cadence_min), key="time_override_cadence")
            if time_format in {"JD_UTC", "MJD_UTC"}:
                convert_to_bjd = st.checkbox("Convert UTC timestamps to BJD_TDB", value=False, key="convert_bjd")
                if convert_to_bjd:
                    st.caption("Uses the target coordinates above and the selected ground-observatory coordinates.")
                    if bool(obs_p.get("space", False)):
                        st.warning("Ground-site barycentric correction requires an Earth location. For spacecraft time stamps, use the mission's calibrated time product rather than a fictitious ground site.")
                    else:
                        time_coord = SkyCoord(float(target_ra_deg) * u.deg, float(target_dec_deg) * u.deg)
                        time_location = EarthLocation.from_geodetic(float(obs_p["lon_deg"]) * u.deg, float(obs_p["lat_deg"]) * u.deg, float(obs_p["height_m"]) * u.m)

        try:
            internal_t, display_offset, display_time_label, is_absolute_time = to_internal_time(
                uploaded_df[time_col].to_numpy(), time_format,
                frame_cadence_min=float(frame_cadence_min), convert_utc_to_bjd=convert_to_bjd,
                coord=time_coord, location=time_location,
            )
            f_raw = pd.to_numeric(uploaded_df[flux_col], errors="coerce").to_numpy(dtype=float)
            e_raw = None if err_choice == "Estimate from robust scatter" else pd.to_numeric(uploaded_df[err_choice], errors="coerce").to_numpy(dtype=float)
            keep = np.ones(len(uploaded_df), dtype=bool)
            if flag_choice != "Do not use a quality flag":
                bad = boolean_bad_row_mask(uploaded_df[flag_choice])
                flagged_rows_removed = int(np.sum(bad))
                keep &= ~bad
            # Manual row exclusions are stored by original DataFrame row index.
            manual_bad = set(int(x) for x in st.session_state.get("manual_excluded_rows", []))
            if manual_bad:
                keep &= ~np.isin(np.arange(len(uploaded_df)), list(manual_bad))
            internal_t = internal_t[keep]
            f_raw = f_raw[keep]
            if e_raw is not None:
                e_raw = e_raw[keep]
            t_obs, raw_flux, err_array, quality_report = prepare_lightcurve(internal_t, f_raw, e_raw, clip_positive_spikes=False)
            source_label = uploaded_file.name
        except Exception as exc:
            st.error(f"Data preparation failed: {exc}")
            st.stop()

    # Reference/injection ephemeris uses the same display convention as the data.
    default_abs_t0 = float(st.session_state.get("archive_t0_bjd", np.nan))
    if is_absolute_time and not np.isfinite(default_abs_t0):
        default_abs_t0 = float(display_offset + np.min(t_obs) + 0.25 * period_days)
    elif not is_absolute_time:
        default_abs_t0 = float(np.min(t_obs) + 0.25 * period_days)

    st.sidebar.subheader("🕒 Reference Ephemeris")
    if is_absolute_time:
        t0_display = st.sidebar.number_input(f"Reference T0 ({display_time_label})", value=float(default_abs_t0), format="%.9f", key="ref_t0_abs")
        reference_t0_internal = float(t0_display - display_offset)
    else:
        reference_t0_internal = st.sidebar.number_input("Reference T0 (elapsed days)", value=float(default_abs_t0), format="%.9f", key="ref_t0_rel")

    if inject_signal:
        st.sidebar.success("💉 Injection mode active — the synthetic transit is multiplied into the uploaded light curve.")
        injected = engine.generate_light_curve(
            t_obs, t0=float(reference_t0_internal), ttv_amp_minutes=float(ttv_amp_minutes),
            ttv_period_days=float(ttv_period_days), include_thermal_phase=False,
        )
        raw_flux = raw_flux * injected
    true_model = engine.generate_light_curve(
        t_obs, t0=float(reference_t0_internal), ttv_amp_minutes=float(ttv_amp_minutes),
        ttv_period_days=float(ttv_period_days), include_thermal_phase=bool(include_thermal_phase),
    ) if inject_signal else np.ones_like(t_obs)

# Common baseline/cadence diagnostics.
if len(t_obs) < 10:
    st.error("Too few usable points for VORTEX analysis.")
    st.stop()
span_days = float(np.ptp(t_obs))
cadence_days = float(np.median(np.diff(np.sort(t_obs)))) if len(t_obs) > 1 else np.nan
cadence_minutes = cadence_days * 1440.0 if np.isfinite(cadence_days) else np.nan

# -----------------------------------------------------------------------------
# DETECTION PIPELINE — blind and known modes deliberately differ
# -----------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.header("🔍 DETECTION")
detection_mode = st.sidebar.radio("Period-search mode", ["Known / Injection Recovery", "Blind Search"], key="detection_mode")
if detection_mode == "Blind Search":
    default_pmax = min(max(0.20, 0.8 * span_days), max(span_days / 2.0, 0.20))
    pmin = st.sidebar.number_input("Blind Pmin (days)", min_value=max(3 * cadence_days, 1e-4), value=max(0.05, 5 * cadence_days), format="%.6f", key="blind_pmin")
    pmax_default = max(float(pmin) * 1.2, default_pmax)
    pmax = st.sidebar.number_input("Blind Pmax (days)", min_value=float(pmin) * 1.01, value=float(pmax_default), format="%.6f", key="blind_pmax")
else:
    pmin = max(0.05 * period_days, 0.5 * period_days, 3 * cadence_days)
    pmax = min(2.0 * period_days, max(span_days / 2.0, 2.0 * period_days))
    st.sidebar.caption(f"Reference period: {period_days:.8g} d")

with st.sidebar.expander("Detrending controls", expanded=False):
    use_gp = st.checkbox("Transit-aware Gaussian-process detrending", value=True, key="use_gp")
    gp_max_points = st.number_input("Maximum GP training points", min_value=50, max_value=3000, value=500, step=50, key="gp_max_points")

# 1) Blind: preliminary robust detrend -> first search -> transit-aware GP -> final search.
# 2) Known: mask supplied ephemeris before GP. If <2 cycles, do not claim period recovery.
try:
    if detection_mode == "Blind Search":
        prelim_flux, prelim_trend, prelim_err = rolling_preliminary_detrend(
            t_obs, raw_flux, err_array, window_days=max(0.15, min(span_days / 3.0, float(pmax)))
        )
        first_search = run_period_search(t_obs, prelim_flux, prelim_err, engine, float(pmin), float(pmax))
        candidate_p = float(first_search["period"])
        candidate_t0 = float(first_search["T0"])
        candidate_duration = max(float(first_search.get("duration", engine.transit_duration_days(period_days=candidate_p))), 2 * cadence_days)
        if use_gp:
            cleaned_flux, gp_trend, cleaned_err, gp_kernel = gp_detrend_transit_aware(
                t_obs, raw_flux, err_array, candidate_p, candidate_t0, candidate_duration, max_gp_points=int(gp_max_points)
            )
        else:
            cleaned_flux, gp_trend, cleaned_err, gp_kernel = prelim_flux, prelim_trend, prelim_err, "GP disabled"
        search_result = run_period_search(t_obs, cleaned_flux, cleaned_err, engine, float(pmin), float(pmax))
    else:
        ref_duration = max(engine.transit_duration_days(period_days=float(period_days)), 2 * cadence_days)
        if use_gp:
            cleaned_flux, gp_trend, cleaned_err, gp_kernel = gp_detrend_transit_aware(
                t_obs, raw_flux, err_array, float(period_days), float(reference_t0_internal), ref_duration, max_gp_points=int(gp_max_points)
            )
        else:
            cleaned_flux, gp_trend, cleaned_err = raw_flux.copy(), np.ones_like(raw_flux), err_array.copy()
            gp_kernel = "GP disabled"
        if span_days < 2.0 * float(period_days):
            search_result = known_ephemeris_result(
                t_obs, cleaned_flux, float(period_days), float(reference_t0_internal), ref_duration, engine.rp_rs
            )
        else:
            search_result = run_period_search(t_obs, cleaned_flux, cleaned_err, engine, max(3 * cadence_days, 0.5 * period_days), 2.0 * period_days)
except Exception as exc:
    st.error(f"Period-search pipeline could not construct a valid search for this dataset: {exc}")
    st.info("For a single short night, use Known / Injection Recovery if the reference period is longer than half the observation baseline.")
    st.stop()

recovered_p = float(search_result.get("period", np.nan))
recovered_t0 = float(search_result.get("T0", np.nan))
recovered_duration = float(search_result.get("duration", np.nan))
if not np.isfinite(recovered_duration) or recovered_duration <= 0:
    recovered_duration = max(engine.transit_duration_days(period_days=recovered_p if np.isfinite(recovered_p) else period_days), 2 * cadence_days)

# Candidate list from search spectrum.
period_grid = np.asarray(search_result.get("periods", []), dtype=float)
power_grid = np.asarray(search_result.get("power", []), dtype=float)
candidate_df = extract_period_candidates(period_grid, power_grid, 7)
if candidate_df.empty and np.isfinite(recovered_p):
    candidate_df = pd.DataFrame({"Rank": [1], "Period_Days": [recovered_p], "Power": [search_result.get("SDE", search_result.get("snr", np.nan))]})

# Analysis model for the recovered ephemeris (primary only for transit comparison).
try:
    fit_model_current = engine.generate_light_curve(
        t_obs, t0=recovered_t0, period_days=recovered_p,
        rp_rs=float(search_result.get("rp_rs", engine.rp_rs)) if np.isfinite(float(search_result.get("rp_rs", np.nan))) else engine.rp_rs,
        include_thermal_phase=False,
    )
except Exception:
    fit_model_current = np.ones_like(t_obs)

# -----------------------------------------------------------------------------
# MAIN HUD
# -----------------------------------------------------------------------------

# Two-row dashboard: 3 cards per row.
# This gives each metric enough horizontal space to display units clearly.

# Row 1
c1, c2, c3 = st.columns(3)

c1.metric(
    "Data",
    f"{len(t_obs)} points",
    delta=f"{span_days * 24:.2f} h span",
    delta_color="off",
)

c2.metric(
    "Search Method",
    str(search_result.get("search_method", "—")),
)

c3.metric(
    "Period",
    f"{recovered_p:.2f} days"
    if np.isfinite(recovered_p)
    else "—",
)

# Row 2
c4, c5, c6 = st.columns(3)

c4.metric(
    "T0",
    f"{display_time(recovered_t0, display_offset):.2f} days"
    if np.isfinite(recovered_t0)
    else "—",
)

sde_val = float(search_result.get("SDE", np.nan))
snr_search = float(search_result.get("snr", np.nan))

c5.metric(
    "SDE / BLS SNR",
    f"{sde_val:.2f}"
    if np.isfinite(sde_val)
    else (
        f"{snr_search:.2f}"
        if np.isfinite(snr_search)
        else "—"
    ),
)

thermal_ppm = float(engine.thermal_fp) * 1e6

c6.metric(
    "Thermal Fp/F*",
    f"{thermal_ppm:.0f} ppm"
    if np.isfinite(thermal_ppm)
    else "—",
)

st.caption(
    f"Source: **{source_label}** · cadence ≈ **{cadence_minutes:.3f} min** · "
    f"time display: **{display_time_label}** · GP: **{gp_kernel}**"
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------

tabs = st.tabs([
    "📈 Dashboard", "🔍 Detection & Vetting", "🌗 Phase / TTV", "🎲 Bayesian Retrieval",
    "🛰 Ground vs Space", "🗓 Observing Planner", "💉 Injection Recovery",
    "🌙 Multi-Night", "📸 ETC / Instrument", "🗃 Catalog / Mission", "💾 Data / Project / Export",
])

# === DASHBOARD ================================================================
with tabs[0]:
    st.subheader("Interactive light-curve workspace")

    x_display = t_obs + display_offset
    gp_x, gp_y = break_lines_at_large_gaps(x_display, gp_trend)
    clean_x, clean_y = break_lines_at_large_gaps(x_display, cleaned_flux)
    fit_x, fit_y = break_lines_at_large_gaps(x_display, fit_model_current)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_display, y=raw_flux, mode="markers",
        name="Raw / injected", marker=dict(size=5), opacity=0.55
    ))
    fig.add_trace(go.Scatter(
        x=gp_x, y=gp_y, mode="lines",
        name="GP / baseline trend", connectgaps=False
    ))
    fig.add_trace(go.Scatter(
        x=clean_x, y=clean_y, mode="markers+lines",
        name="Detrended", marker=dict(size=4), line=dict(width=1),
        connectgaps=False
    ))
    if np.any(np.abs(fit_model_current - 1.0) > 1e-10):
        fig.add_trace(go.Scatter(
            x=fit_x, y=fit_y, mode="lines",
            name="Recovered transit model", line=dict(width=2),
            connectgaps=False
        ))

    # For synthetic ground-based observations, show the *same* daytime
    # windows that were actually removed from the time sampling.
    if (
        workflow == "Explore / Simulate"
        and not bool(obs_p.get("space", False))
        and bool(apply_daytime_gaps)
    ):
        add_daytime_gap_shading(
            fig,
            baseline_days=float(sim_baseline),
            night_hours=float(sim_night_hours),
            phase_offset_days=float(window_phase),
            x_offset=float(display_offset),
        )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        xaxis_title=display_time_label,
        yaxis_title="Normalized flux",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if (
        workflow == "Explore / Simulate"
        and not bool(obs_p.get("space", False))
        and bool(apply_daytime_gaps)
    ):
        st.caption(
            "Gray bands are true daytime gaps: those timestamps were removed from the "
            "synthetic dataset before detrending and transit searching."
        )

    residual = cleaned_flux - fit_model_current
    fig_r = go.Figure(go.Scatter(
        x=x_display, y=residual * 1e6, mode="markers",
        marker=dict(size=5), name="Residual"
    ))
    fig_r.add_hline(y=0)

    if (
        workflow == "Explore / Simulate"
        and not bool(obs_p.get("space", False))
        and bool(apply_daytime_gaps)
    ):
        add_daytime_gap_shading(
            fig_r,
            baseline_days=float(sim_baseline),
            night_hours=float(sim_night_hours),
            phase_offset_days=float(window_phase),
            x_offset=float(display_offset),
            label_first=False,
        )

    fig_r.update_layout(
        template="plotly_dark",
        height=260,
        xaxis_title=display_time_label,
        yaxis_title="Residual (ppm)",
    )
    st.plotly_chart(fig_r, use_container_width=True)

    d_raw = estimate_depth_at_ephemeris(t_obs, raw_flux, err_array, recovered_t0, recovered_p, recovered_duration)
    d_clean = estimate_depth_at_ephemeris(t_obs, cleaned_flux, cleaned_err, recovered_t0, recovered_p, recovered_duration)
    a, b, c, d = st.columns(4)
    a.metric("Raw depth", f"{d_raw['depth']*1e6:.0f} ppm" if np.isfinite(d_raw['depth']) else "—")
    b.metric("Detrended depth", f"{d_clean['depth']*1e6:.0f} ppm" if np.isfinite(d_clean['depth']) else "—")
    if np.isfinite(d_raw["depth"]) and d_raw["depth"] != 0 and np.isfinite(d_clean["depth"]):
        gp_change = 100 * (d_clean["depth"] - d_raw["depth"]) / abs(d_raw["depth"])
        c.metric("GP depth change", f"{gp_change:+.1f}%")
        if abs(gp_change) > 10:
            st.warning("The detrending step changes the estimated transit depth by >10%. Inspect the GP mask/model before interpreting the radius ratio.")
    else:
        c.metric("GP depth change", "—")
    d.metric("Phase coverage", f"{100*phase_coverage_fraction(t_obs, recovered_p):.1f}%" if np.isfinite(recovered_p) else "—")

    with st.expander("Current system geometry / model assumptions"):
        geom = pd.DataFrame({
            "Quantity": ["a/R_host", "Rp/Rhost", "Impact parameter b", "Approx. T14 (min)", "Host mass (M_sun)", "Host radius (R_sun)"],
            "Value": [engine.a_scaled(), engine.rp_rs, engine.impact_parameter(), engine.transit_duration_days()*1440.0, engine.host_mass_solar, engine.host_radius_solar],
        })
        st.dataframe(geom, use_container_width=True, hide_index=True)
        st.caption("T14 is an analytic planning/masking approximation. BATMAN is used for the actual light-curve model.")

# === DETECTION & VETTING =======================================================
with tabs[1]:
    st.subheader("Period search and candidate vetting")
    st.write(search_result.get("search_note", ""))
    if len(period_grid) > 0 and len(power_grid) == len(period_grid):
        pf = go.Figure(go.Scatter(x=period_grid, y=power_grid, mode="lines", name="Search statistic"))
        pf.add_vline(x=recovered_p, line_dash="dash")
        if detection_mode == "Known / Injection Recovery":
            pf.add_vline(x=period_days, line_dash="dot")
        pf.update_layout(template="plotly_dark", height=420, xaxis_title="Trial period (days)", yaxis_title="TLS SDE / BLS depth SNR")
        st.plotly_chart(pf, use_container_width=True)
    else:
        st.info("No independent periodogram is shown because the baseline cannot identify multiple cycles of the supplied reference period.")

    st.markdown("#### Ranked period candidates")
    st.dataframe(candidate_df, use_container_width=True, hide_index=True)
    if not candidate_df.empty:
        candidate_rank = st.selectbox("Candidate to inspect", candidate_df["Rank"].tolist(), key="vet_candidate_rank")
        cand_period = float(candidate_df.loc[candidate_df["Rank"] == candidate_rank, "Period_Days"].iloc[0])
        try:
            vet = refine_and_vet_candidate(t_obs, cleaned_flux, cleaned_err, cand_period, recovered_duration)
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Refined period", f"{vet['period']:.8f} d")
            v2.metric("Depth SNR", f"{vet['depth_snr']:.2f}")
            v3.metric("Odd/even mismatch", f"{vet['odd_even_sigma']:.2f} σ" if np.isfinite(vet['odd_even_sigma']) else "—")
            v4.metric("Secondary-like SNR", f"{vet['secondary_snr']:.2f}" if np.isfinite(vet['secondary_snr']) else "—")

            vet_table = pd.DataFrame({
                "Diagnostic": ["Observed transit centers", "Odd/even depth mismatch", "Secondary-like depth SNR", "Half-period depth"],
                "Value": [vet["n_transits"], vet["odd_even_sigma"], vet["secondary_snr"], vet["half_period_depth"]],
                "Interpretation": [
                    "More events generally improve periodic confidence.",
                    "Large mismatch can indicate an eclipsing binary / systematics; not a standalone rejection.",
                    "A significant secondary-like event warrants astrophysical vetting.",
                    "Strong half-period structure can indicate an alias.",
                ],
            })
            st.dataframe(vet_table, use_container_width=True, hide_index=True)

            phase_c = phase_from_ephemeris(t_obs, vet["t0"], vet["period"]) / vet["period"]
            order = np.argsort(phase_c)
            fph = go.Figure(go.Scatter(x=phase_c[order], y=cleaned_flux[order], mode="markers", marker=dict(size=5), name="Data"))
            fph.update_layout(template="plotly_dark", height=350, xaxis_title="Orbital phase", yaxis_title="Detrended flux")
            st.plotly_chart(fph, use_container_width=True)
        except Exception as exc:
            st.warning(f"Candidate vetting could not be completed: {exc}")

# === PHASE/TTV ================================================================
with tabs[2]:
    st.subheader("Phase-folding, ephemeris, thermal eclipse and injected TTVs")
    phase_time = phase_from_ephemeris(t_obs, recovered_t0, recovered_p)
    phase_bins = weighted_phase_bins(t_obs, cleaned_flux, cleaned_err, recovered_t0, recovered_p, n_bins=35)
    pfold = go.Figure()
    pfold.add_trace(go.Scatter(x=phase_time * 24.0, y=cleaned_flux, mode="markers", opacity=0.35, marker=dict(size=5), name="Data"))
    if not phase_bins.empty:
        pfold.add_trace(go.Scatter(
            x=phase_bins["Phase_Days"] * 24.0, y=phase_bins["Flux"],
            error_y=dict(type="data", array=phase_bins["Error"], visible=True),
            mode="markers", marker=dict(size=8), name="Weighted bins"
        ))
    tmod = np.linspace(-0.5 * recovered_p, 0.5 * recovered_p, 1000) + recovered_t0
    try:
        mmod = engine.generate_light_curve(tmod, t0=recovered_t0, period_days=recovered_p, include_thermal_phase=False)
        pfold.add_trace(go.Scatter(x=(tmod - recovered_t0) * 24.0, y=mmod, mode="lines", name="Transit model"))
    except Exception:
        pass
    pfold.update_layout(template="plotly_dark", height=430, xaxis_title="Hours from mid-transit", yaxis_title="Normalized flux")
    st.plotly_chart(pfold, use_container_width=True)

    colp, cole = st.columns(2)
    with colp:
        st.markdown("#### Secondary eclipse / thermal phase")
        try:
            sec = engine.secondary_time(recovered_t0, recovered_p)
            ts = np.linspace(sec - 0.15 * recovered_p, sec + 0.15 * recovered_p, 500)
            fs = engine.generate_light_curve(ts, t0=recovered_t0, period_days=recovered_p, include_thermal_phase=True)
            sp = go.Figure(go.Scatter(x=(ts - sec) * 24.0, y=fs, mode="lines"))
            sp.update_layout(template="plotly_dark", height=320, xaxis_title="Hours from secondary conjunction", yaxis_title="Total normalized flux")
            st.plotly_chart(sp, use_container_width=True)
            st.caption(f"Band-integrated blackbody Fp/F* ≈ {engine.thermal_fp:.3e}. Treat this as an approximation, especially for cool substellar atmospheres.")
        except Exception as exc:
            st.info(f"Secondary timing unavailable for this geometry: {exc}")
    with cole:
        st.markdown("#### Upcoming ephemeris")
        epochs = np.arange(0, 8)
        eph = pd.DataFrame({
            "Epoch": epochs,
            f"Mid-transit ({display_time_label})": [display_time(recovered_t0 + e * recovered_p, display_offset) for e in epochs],
        })
        st.dataframe(eph, use_container_width=True, hide_index=True)

    if ttv_amp_minutes > 0:
        st.markdown("#### Injected TTV O−C")
        ttv_df = make_ttv_table(float(np.min(t_obs)), float(np.max(t_obs)), recovered_t0, recovered_p, float(ttv_amp_minutes), float(ttv_period_days))
        oc = go.Figure(go.Scatter(x=ttv_df["Epoch"], y=ttv_df["TTV_Minutes"], mode="lines+markers"))
        oc.update_layout(template="plotly_dark", height=300, yaxis_title="O−C (min)")
        st.plotly_chart(oc, use_container_width=True)
        st.dataframe(ttv_df, use_container_width=True, hide_index=True)

# === BAYESIAN RETRIEVAL ========================================================
with tabs[3]:
    st.subheader("Bayesian transit retrieval")
    st.caption("MCMC uses a stateless BATMAN likelihood with impact parameter, T0, period, additive baseline offset, and white-jitter term. Limb darkening can optionally be sampled with Kipping q1/q2 parameters.")
    m1, m2, m3, m4 = st.columns(4)
    n_steps = m1.number_input("Steps", 300, 20000, 1500, 100, key="mcmc_steps")
    burn_frac = m2.slider("Burn-in fraction", 0.05, 0.60, 0.25, 0.05, key="mcmc_burn_frac")
    n_walkers = m3.selectbox("Walkers", [24, 32, 48, 64, 96], index=2, key="mcmc_walkers")
    fit_ld = m4.checkbox("Fit limb darkening", value=False, key="mcmc_fit_ld")
    ld_sigma = st.number_input("Gaussian σ on u1/u2 when fitting limb darkening", min_value=0.001, value=0.10, step=0.01, key="mcmc_ld_sigma") if fit_ld else 0.0

    if st.button("🚀 Run Bayesian retrieval", key="run_mcmc_pro", type="primary"):
        try:
            bounds = advanced_bounds_from_result(search_result, engine)
            pos = initialize_advanced_walkers(int(n_walkers), search_result, engine, bounds, cleaned_err, bool(fit_ld), seed=2026)
            ld_prior = (engine.u1, float(ld_sigma), engine.u2, float(ld_sigma)) if fit_ld else None
            sampler = emcee.EnsembleSampler(
                int(n_walkers), pos.shape[1], advanced_log_posterior,
                args=(t_obs, cleaned_flux, cleaned_err, engine, bounds, bool(fit_ld), ld_prior),
            )
            sampler.run_mcmc(pos, int(n_steps), progress=False)
            burn = min(int(round(n_steps * burn_frac)), int(n_steps) - 10)
            chain = sampler.get_chain()
            flat = sampler.get_chain(discard=burn, thin=max(1, int((n_steps - burn) / 500)), flat=True)
            names = ["Rp/Rs", "b", "T0", "Period", "log_jitter", "offset"] + (["q1", "q2"] if fit_ld else [])
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except Exception:
                tau = np.full(pos.shape[1], np.nan)
            st.session_state.mcmc_pro = {"chain": chain, "flat": flat, "names": names}
            st.session_state.mcmc_meta = {
                "burn": burn, "acceptance": float(np.mean(sampler.acceptance_fraction)),
                "tau": np.asarray(tau, dtype=float), "fit_ld": bool(fit_ld),
            }
        except Exception as exc:
            st.error(f"MCMC failed: {exc}")

    if isinstance(st.session_state.mcmc_pro, dict):
        mc = st.session_state.mcmc_pro
        meta = st.session_state.mcmc_meta
        flat = np.asarray(mc["flat"])
        names = list(mc["names"])
        if flat.size:
            summary = posterior_summary(flat, names)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            aa, tt, nn = st.columns(3)
            aa.metric("Mean acceptance", f"{meta['acceptance']:.3f}")
            finite_tau = np.asarray(meta["tau"])[np.isfinite(meta["tau"])]
            tt.metric("Max autocorr time", f"{np.max(finite_tau):.1f}" if len(finite_tau) else "not converged")
            nn.metric("Posterior samples", f"{len(flat):,}")
            if not (0.15 <= meta["acceptance"] <= 0.65):
                st.warning("Acceptance fraction is outside a common practical range. Inspect traces and increase chain length / adjust priors before scientific interpretation.")

            col_trace, col_corner = st.columns(2)
            with col_trace:
                trfig = make_trace_plot(np.asarray(mc["chain"]), names)
                st.pyplot(trfig)
                plt.close(trfig)
            with col_corner:
                truth = None
                if workflow in {"Explore / Simulate", "Inject & Recover"}:
                    truth = {"Rp/Rs": engine.rp_rs, "T0": reference_t0_internal, "Period": period_days, "b": engine.impact_parameter()}
                cfig = make_corner_plot(flat, names, truth=truth)
                st.pyplot(cfig)
                plt.close(cfig)

            med = {name: float(np.median(flat[:, i])) for i, name in enumerate(names)}
            ic = mcmc_information_criteria(t_obs, cleaned_flux, cleaned_err, engine, med, bool(meta["fit_ld"]))
            st.markdown("#### Quick transit-vs-null BIC diagnostic")
            st.dataframe(pd.DataFrame([ic]), use_container_width=True, hide_index=True)
            st.caption("BIC is a model-selection diagnostic, not a false-alarm probability or planet-validation probability.")

            posterior_df = pd.DataFrame(flat, columns=names)
            st.download_button("Download posterior samples CSV", posterior_df.to_csv(index=False), "vortex_posterior_samples.csv", "text/csv")

# === GROUND VS SPACE ===========================================================
with tabs[4]:
    st.subheader("MMT vs Lazuli: window-function science case")
    st.caption(
        "The comparison can isolate the sampling-window effect or use each profile's illustrative noise assumptions. "
        "Lazuli's 24-hour profile here is an idealized continuous-coverage benchmark, not a mission pointing simulator."
    )
    ca, cb, cc, cd = st.columns(4)
    cmp_baseline = ca.number_input("Comparison baseline (days)", 0.5, 30.0, max(3.0, 4 * period_days), 0.5, key="cmp_baseline")
    cmp_cadence = cb.number_input("Common cadence (min)", 0.1, 60.0, 5.0, 0.5, key="cmp_cadence")
    cmp_realism = cc.selectbox("Comparison", ["Window function only", "Profile noise + window"], key="cmp_mode")
    cmp_trials = cd.number_input("Random phase trials", 5, 200, 30, 5, key="cmp_trials")

    if st.button("Run MMT ↔ Lazuli comparison", key="run_compare", type="primary"):
        try:
            profiles = {
                "MMT / MMIRS": ObservatoryProfiles.get_profile("MMT / MMIRS (Ground)"),
                "Lazuli 3-m": ObservatoryProfiles.get_profile("Lazuli 3-m (Space; idealized continuous benchmark)"),
            }
            common_white = 500.0
            common_red = 300.0
            rows, payload = [], {}
            for j, (label, prof) in enumerate(profiles.items()):
                wp = common_white if cmp_realism == "Window function only" else float(prof["white_noise_ppm"])
                rp = common_red if cmp_realism == "Window function only" else float(prof["red_noise_ppm"])
                tcmp, fcmp, ecmp, mcmp = simulate_profile_observation(
                    engine, prof, t0=0.25 * period_days, baseline_days=float(cmp_baseline), cadence_min=float(cmp_cadence),
                    white_ppm=wp, red_ppm=rp, red_tau_hours=2.0, seed=400 + j,
                )
                dur = max(engine.transit_duration_days(), 2 * cmp_cadence / 1440.0)
                cov = transit_coverage_metrics(tcmp, 0.25 * period_days, period_days, dur)
                phasecov = phase_coverage_fraction(tcmp, period_days)
                pcmp_min = max(3*np.median(np.diff(np.sort(tcmp))), 0.5 * period_days)
                pcmp_max = min(2 * period_days, np.ptp(tcmp) / 2)
                r = fast_bls_candidate(tcmp, fcmp, ecmp, pcmp_min, pcmp_max, dur)
                try:
                    headline_search = run_period_search(tcmp, fcmp, ecmp, engine, pcmp_min, pcmp_max)
                except Exception:
                    headline_search = {"period": r["period"], "snr": r["snr"], "SDE": np.nan, "FAP": np.nan, "search_method": "BLS"}
                pwin, wwin = spectral_window(tcmp, max(0.1, 0.25 * period_days), min(max(2 * period_days, 0.2), max(np.ptp(tcmp) / 2, 0.3)))

                success = 0
                rngcmp = np.random.default_rng(9000 + j)
                for k in range(int(cmp_trials)):
                    phase0 = float(rngcmp.uniform(0, period_days))
                    tx, fx, ex, _ = simulate_profile_observation(
                        engine, prof, t0=phase0, baseline_days=float(cmp_baseline), cadence_min=float(cmp_cadence),
                        white_ppm=wp, red_ppm=rp, red_tau_hours=2.0, seed=10000 + 100 * j + k,
                    )
                    rr = fast_bls_candidate(tx, fx, ex, max(0.1, 0.5 * period_days), min(2 * period_days, np.ptp(tx) / 2), dur)
                    success += int(period_recovered(rr["period"], period_days, relative_tolerance=0.02, allow_harmonics=True) and np.isfinite(rr["snr"]) and rr["snr"] >= 5)

                rows.append({
                    "Observatory": label, "N_points": len(tcmp), "Phase_coverage": phasecov,
                    "Predicted_transits": cov["n_predicted"], "Any_coverage": cov["n_any"], ">=50%_covered": cov["n_half"],
                    "Mean_transit_coverage": cov["mean_fraction"],
                    "Search_method": headline_search.get("search_method", "BLS"),
                    "Recovered_period_d": float(headline_search.get("period", r["period"])),
                    "TLS_SDE": float(headline_search.get("SDE", np.nan)),
                    "TLS_FAP": float(headline_search.get("FAP", np.nan)),
                    "BLS_depth_SNR": r["snr"],
                    "Random_phase_recovery_fraction": success / int(cmp_trials),
                })
                payload[label] = {"t": tcmp, "f": fcmp, "m": mcmp, "pwin": pwin, "wwin": wwin}
            st.session_state.compare_result = {"table": pd.DataFrame(rows), "payload": payload}
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    if isinstance(st.session_state.compare_result, dict):
        ct = st.session_state.compare_result["table"]
        st.dataframe(ct.style.format({"Phase_coverage": "{:.1%}", "Mean_transit_coverage": "{:.1%}", "Random_phase_recovery_fraction": "{:.1%}"}), use_container_width=True, hide_index=True)
        payload = st.session_state.compare_result["payload"]
        g1, g2 = st.columns(2)
        for col, label in zip([g1, g2], ["MMT / MMIRS", "Lazuli 3-m"]):
            with col:
                p = payload[label]
                mx, my = break_lines_at_large_gaps(p["t"], p["m"])

                fcmp = go.Figure()
                fcmp.add_trace(go.Scatter(
                    x=p["t"], y=p["f"], mode="markers",
                    marker=dict(size=4), name=label
                ))
                fcmp.add_trace(go.Scatter(
                    x=mx, y=my, mode="lines",
                    name="Injected model", connectgaps=False
                ))

                if label == "MMT / MMIRS":
                    mmt_profile = ObservatoryProfiles.get_profile("MMT / MMIRS (Ground)")
                    add_daytime_gap_shading(
                        fcmp,
                        baseline_days=float(np.max(p["t"]) + np.median(np.diff(p["t"])) if len(p["t"]) > 1 else np.max(p["t"])),
                        night_hours=float(mmt_profile["night_hours"]),
                        phase_offset_days=0.0,
                    )

                fcmp.update_layout(
                    template="plotly_dark",
                    height=300,
                    title=label,
                    xaxis_title="Elapsed days",
                    yaxis_title="Flux",
                )
                st.plotly_chart(fcmp, use_container_width=True)

                if label == "MMT / MMIRS":
                    st.caption(
                        "Gray regions are the idealized daytime intervals removed from the MMT sampling."
                    )

                fw = go.Figure(go.Scatter(x=p["pwin"], y=p["wwin"], mode="lines"))
                fw.update_layout(
                    template="plotly_dark",
                    height=260,
                    title="Sampling spectral window",
                    xaxis_title="Period (d)",
                    yaxis_title="Normalized window power",
                )
                st.plotly_chart(fw, use_container_width=True)

# === OBSERVING PLANNER =========================================================
with tabs[5]:
    st.subheader("Ground-based transit observing planner")
    st.caption("Date-specific altitude, airmass, astronomical-night, Moon-separation and contiguous pre/post-baseline checks. Spacecraft pointing constraints are mission-specific and are intentionally not approximated here.")
    planner_site = st.selectbox("Ground observatory", ["MMT Observatory", "Custom ground site"], key="planner_site")
    if planner_site == "MMT Observatory":
        pl_lat, pl_lon, pl_h = 31.6887778, -110.8845556, 2606.0
    else:
        pa, pb, pc = st.columns(3)
        pl_lat = pa.number_input("Latitude", -90.0, 90.0, 31.6888, format="%.6f", key="planner_lat")
        pl_lon = pb.number_input("Longitude east+", -180.0, 180.0, -110.8846, format="%.6f", key="planner_lon")
        pl_h = pc.number_input("Elevation (m)", -500.0, 10000.0, 2000.0, key="planner_height")

    pa, pb, pc, pd_ = st.columns(4)
    planner_t0 = pa.number_input(
        "Ephemeris T0 (BJD_TDB)",
        value=float(st.session_state.archive_t0_bjd) if np.isfinite(float(st.session_state.archive_t0_bjd)) else 2461294.5,
        format="%.9f", key="planner_t0"
    )
    planner_period = pb.number_input("Period (d)", min_value=1e-5, value=float(period_days), format="%.9f", key="planner_period")
    planner_start = pc.date_input("Start date", value=date.today(), key="planner_start")
    planner_end = pd_.date_input("End date", value=date.today() + timedelta(days=30), key="planner_end")
    ca, cb, cc, cd, ce = st.columns(5)
    min_alt = ca.number_input("Min altitude (deg)", 0.0, 90.0, 30.0, 1.0, key="planner_min_alt")
    max_airmass = cb.number_input("Max airmass", 1.0, 5.0, 2.0, 0.1, key="planner_airmass")
    moon_sep = cc.number_input("Min Moon sep. (deg)", 0.0, 180.0, 30.0, 5.0, key="planner_moon")
    pre_hr = cd.number_input("Required pre-baseline (h)", 0.0, 8.0, 1.0, 0.25, key="planner_pre")
    post_hr = ce.number_input("Required post-baseline (h)", 0.0, 8.0, 1.0, 0.25, key="planner_post")

    if st.button("Calculate observing windows", key="run_planner", type="primary"):
        try:
            coord = SkyCoord(float(target_ra_deg) * u.deg, float(target_dec_deg) * u.deg)
            loc = EarthLocation.from_geodetic(float(pl_lon) * u.deg, float(pl_lat) * u.deg, float(pl_h) * u.m)
            duration_plan = engine.transit_duration_days(period_days=float(planner_period))
            if duration_plan <= 0:
                raise ValueError("The current geometry does not produce a transit, so no transit window can be planned.")
            table = build_observing_plan(
                float(planner_t0), float(planner_period), duration_plan, coord, loc,
                planner_start, planner_end, float(min_alt), float(max_airmass), float(moon_sep), float(pre_hr), float(post_hr),
            )
            st.session_state.planner_result = table
        except Exception as exc:
            st.error(f"Planner failed: {exc}")

    if isinstance(st.session_state.planner_result, pd.DataFrame):
        plan = st.session_state.planner_result.copy()
        if plan.empty:
            st.info("No transit centers fall in the requested date interval.")
        else:
            show = plan.rename(columns={
                "tc_bjd": "BJD_TDB", "tc_utc": "UTC mid-transit", "event_coverage": "Transit coverage",
                "pre_baseline_hr": "Pre baseline (h)", "post_baseline_hr": "Post baseline (h)",
                "altitude_mid_deg": "Alt mid (deg)", "airmass_mid": "Airmass mid", "moon_sep_mid_deg": "Moon sep (deg)",
                "sun_alt_mid_deg": "Sun alt (deg)", "score": "Score /100",
            })
            st.dataframe(show.style.format({"Transit coverage": "{:.0%}", "Score /100": "{:.1f}"}), use_container_width=True, hide_index=True)
            ps = go.Figure(go.Bar(x=show["UTC mid-transit"], y=show["Score /100"], text=show["Transit coverage"].map(lambda x: f"{x:.0%}")))
            ps.update_layout(template="plotly_dark", height=320, yaxis_range=[0, 100], yaxis_title="Observability score", xaxis_title="UTC mid-transit")
            st.plotly_chart(ps, use_container_width=True)

# === INJECTION RECOVERY ========================================================
with tabs[6]:
    st.subheader("Injection–recovery completeness")
    st.caption("Signals are injected into the current detrended cadence/noise realization and recovered with BLS. A cell is only evaluated when the baseline can contain at least two periods.")
    ia, ib, ic, idd = st.columns(4)
    p_lo = ia.number_input("Grid Pmin (d)", min_value=max(3*cadence_days, 0.001), value=max(0.05, 6*cadence_days), format="%.5f", key="inj_pmin")
    p_hi = ib.number_input("Grid Pmax (d)", min_value=float(p_lo)*1.02, value=max(float(p_lo)*1.5, min(span_days/2, max(period_days*2, 0.5))), format="%.5f", key="inj_pmax")
    r_lo = ic.number_input("Radius min (R_Earth)", min_value=0.05, value=0.5, step=0.25, key="inj_rmin")
    r_hi = idd.number_input("Radius max (R_Earth)", min_value=float(r_lo)+0.01, value=max(5.0, companion_radius_earth*2), step=0.5, key="inj_rmax")
    ja, jb, jc, jd = st.columns(4)
    n_p = ja.slider("Period grid points", 3, 15, 6, key="inj_np")
    n_r = jb.slider("Radius grid points", 3, 15, 6, key="inj_nr")
    trials = jc.slider("Trials / cell", 3, 100, 10, key="inj_trials")
    threshold = jd.number_input("BLS depth-SNR threshold", 2.0, 20.0, 5.0, 0.5, key="inj_threshold")

    if st.button("Run injection–recovery grid", key="run_injrec", type="primary"):
        try:
            periods = np.geomspace(float(p_lo), float(p_hi), int(n_p))
            radii = np.geomspace(float(r_lo), float(r_hi), int(n_r))
            comp, detail = injection_recovery_grid(
                t_obs, cleaned_flux, cleaned_err, engine, periods, radii, int(trials), float(threshold), 0.02, True, 314159
            )
            st.session_state.injrec_result = comp
            st.session_state.injrec_detail = detail
        except Exception as exc:
            st.error(f"Injection–recovery failed: {exc}")

    if isinstance(st.session_state.injrec_result, pd.DataFrame):
        comp = st.session_state.injrec_result
        pivot = comp.pivot(index="Radius_REarth", columns="Period_Days", values="Completeness").sort_index(ascending=True)
        hm = go.Figure(go.Heatmap(x=pivot.columns, y=pivot.index, z=pivot.values, zmin=0, zmax=1, colorbar_title="Completeness"))
        hm.update_layout(template="plotly_dark", height=450, xaxis_title="Injected period (d)", yaxis_title="Injected radius (R_Earth)")
        st.plotly_chart(hm, use_container_width=True)
        st.dataframe(comp.style.format({"Completeness":"{:.1%}", "Median_SNR":"{:.2f}"}), use_container_width=True, hide_index=True)
        if isinstance(st.session_state.injrec_detail, pd.DataFrame) and not st.session_state.injrec_detail.empty:
            st.download_button("Download injection trial table", st.session_state.injrec_detail.to_csv(index=False), "vortex_injection_recovery_trials.csv", "text/csv")

# === MULTI NIGHT ===============================================================
with tabs[7]:
    st.subheader("Multi-night / multi-file analysis")
    st.caption("Each night is normalized and detrended independently; shared astrophysical ephemeris is then evaluated across all nights. For frame-index files, you must specify each night's time offset if you want a physically meaningful joint period across separated nights.")
    nights = st.file_uploader("Upload multiple CSV light curves", type=["csv"], accept_multiple_files=True, key="multi_files")
    if nights:
        first = pd.read_csv(nights[0])
        mcols = first.columns.tolist()
        ma, mb, mc, md = st.columns(4)
        mtcol = ma.selectbox("Time column", mcols, index=0, key="multi_time_col")
        mfcol = mb.selectbox("Flux column", mcols, index=default_flux_column_index(mcols), key="multi_flux_col")
        merr_opts = ["Estimate from robust scatter"] + mcols
        mecol = mc.selectbox("Error column", merr_opts, index=0, key="multi_err_col")
        mflag_opts = ["Do not use a quality flag"] + mcols
        mf_default = next((i for i,c in enumerate(mflag_opts) if str(c).lower().replace("_","") == "outlierremoved"), 0)
        mflag = md.selectbox("Quality flag", mflag_opts, index=mf_default, key="multi_flag")
        guessed_mfmt = guess_time_format_from_column(mtcol, first[mtcol].to_numpy())
        multi_time_options = ["Frame Number", "Elapsed Minutes", "Elapsed Hours", "Elapsed Days", "BJD_TDB", "JD_UTC", "MJD_UTC"]
        mca, mcb, mcc = st.columns(3)
        mfmt = mca.selectbox(
            "Time interpretation",
            multi_time_options,
            index=multi_time_options.index(guessed_mfmt),
            key="multi_time_format",
            help="Name-based default only. VORTEX does not infer BJD/JD/MJD from the numerical magnitude alone.",
        )
        mframe_cad = mcb.number_input("Frame cadence (min; if frame time)", min_value=0.001, value=float(obs_p["cadence_min"]), step=0.1, key="multi_cad")
        use_filename_dates = mcc.checkbox("Use YYYY.MMDD / YYYY-MM-DD filename dates as night offsets", value=True, key="multi_dates")

        # A joint ephemeris must live on exactly the same time axis as the uploaded nights.
        # Do not reuse an internal T0 from another dataset after subtracting a different
        # absolute-time origin; that would silently shift every transit window.
        first_time_numeric = pd.to_numeric(first[mtcol], errors="coerce").to_numpy(dtype=float)
        first_time_finite = first_time_numeric[np.isfinite(first_time_numeric)]
        if first_time_finite.size == 0:
            st.error("The selected multi-night time column has no finite values in the first file.")
            multi_t0_display_default = float(reference_t0_internal)
        elif mfmt == "BJD_TDB":
            archive_t0 = float(st.session_state.get("archive_t0_bjd", np.nan))
            if np.isfinite(archive_t0):
                multi_t0_display_default = archive_t0
            elif is_absolute_time and display_time_label == "BJD_TDB":
                multi_t0_display_default = float(reference_t0_internal + display_offset)
            else:
                multi_t0_display_default = float(np.nanmin(first_time_finite) + 0.25 * period_days)
        elif mfmt in {"JD_UTC", "MJD_UTC"}:
            # A BJD_TDB epoch cannot be silently reused as JD/MJD UTC. The user can
            # enter the epoch in the selected time system; this default is only local.
            multi_t0_display_default = float(np.nanmin(first_time_finite) + 0.25 * period_days)
        else:
            multi_t0_display_default = float(reference_t0_internal)

        if mfmt in {"BJD_TDB", "JD_UTC", "MJD_UTC"}:
            multi_t0_display = st.number_input(
                f"Reference T0 in {mfmt}",
                value=float(multi_t0_display_default),
                format="%.9f",
                key="multi_t0_absolute",
                help="Enter T0 in the same time system used by the multi-night files. For precision TTV work, BJD_TDB is recommended.",
            )
        else:
            multi_t0_display = st.number_input(
                "Reference T0 on combined elapsed-day axis",
                value=float(multi_t0_display_default),
                format="%.9f",
                key="multi_t0_relative",
                help="For frame/elapsed files with filename date offsets, day 0 is the earliest recognized observing date.",
            )

        if st.button("Process multi-night dataset", key="run_multi", type="primary"):
            try:
                rows = []
                all_t, all_f, all_e = [], [], []
                dates = [infer_date_from_filename(f.name) for f in nights]
                dated = [d for d in dates if d is not None]
                date0 = min(dated) if dated and use_filename_dates else None

                # Absolute-time files must share one common numerical origin.
                # Subtracting a separate integer offset from every file would destroy
                # the inter-night separation and therefore the joint period information.
                absolute_formats = {"BJD_TDB", "JD_UTC", "MJD_UTC"}
                common_abs_origin = None
                if mfmt in absolute_formats:
                    first_values = pd.to_numeric(first[mtcol], errors="coerce").to_numpy(dtype=float)
                    finite_first = first_values[np.isfinite(first_values)]
                    if finite_first.size == 0:
                        raise ValueError("The selected multi-night time column has no finite values.")
                    # MJD and JD/BJD use different numerical origins; we keep the selected
                    # system untouched and subtract one common integer origin only for
                    # floating-point stability.
                    common_abs_origin = float(np.floor(np.nanmin(finite_first)))
                    multi_ref_t0 = float(multi_t0_display) - common_abs_origin
                else:
                    multi_ref_t0 = float(multi_t0_display)

                for uf, dte in zip(nights, dates):
                    df = pd.read_csv(uf)
                    missing = [c for c in [mtcol, mfcol] if c not in df.columns]
                    if missing:
                        raise ValueError(f"{uf.name} is missing mapped columns: {missing}")

                    if mfmt in absolute_formats:
                        raw_tn = pd.to_numeric(df[mtcol], errors="coerce").to_numpy(dtype=float)
                        raw_fn = pd.to_numeric(df[mfcol], errors="coerce").to_numpy(dtype=float)
                        raw_en = None if mecol == "Estimate from robust scatter" else pd.to_numeric(df[mecol], errors="coerce").to_numpy(dtype=float)
                        keep_n = np.ones(len(df), dtype=bool)
                        if mflag != "Do not use a quality flag" and mflag in df.columns:
                            keep_n &= ~boolean_bad_row_mask(df[mflag])
                        t_n = raw_tn[keep_n] - float(common_abs_origin)
                        f_use = raw_fn[keep_n]
                        e_use = raw_en[keep_n] if raw_en is not None else None
                        t_n, f_n, e_n, rep = prepare_lightcurve(t_n, f_use, e_use, clip_positive_spikes=False)
                    else:
                        t_n, f_n, e_n, rep = process_night_dataframe(df, mtcol, mfcol, mecol, mflag, mfmt, float(mframe_cad))
                        if date0 is not None and dte is not None:
                            t_n = t_n + float((dte - date0).days)
                    # Night-level detrending uses the supplied/recovered common ephemeris.
                    dur_n = max(engine.transit_duration_days(period_days=period_days), 2*np.median(np.diff(np.sort(t_n))))
                    f_c, tr_n, e_c, _ = gp_detrend_transit_aware(t_n, f_n, e_n, period_days, multi_ref_t0, dur_n, max_gp_points=500)
                    dep = estimate_depth_at_ephemeris(t_n, f_c, e_c, multi_ref_t0, period_days, dur_n)
                    rows.append({"File": uf.name, "Date": str(dte) if dte else "", "N": len(t_n), "Baseline_h": np.ptp(t_n)*24, "Depth_ppm": dep["depth"]*1e6 if np.isfinite(dep["depth"]) else np.nan, "Depth_err_ppm": dep["error"]*1e6 if np.isfinite(dep["error"]) else np.nan})
                    all_t.append(t_n); all_f.append(f_c); all_e.append(e_c)
                ta, fa, ea = np.concatenate(all_t), np.concatenate(all_f), np.concatenate(all_e)
                order = np.argsort(ta); ta,fa,ea=ta[order],fa[order],ea[order]
                joint = run_period_search(ta,fa,ea,engine,max(3*np.median(np.diff(np.unique(ta))),0.5*period_days),min(2*period_days,np.ptp(ta)/2)) if np.ptp(ta)>=2*period_days else known_ephemeris_result(ta,fa,period_days,multi_ref_t0,engine.transit_duration_days(),engine.rp_rs)
                st.session_state.multi_result = {"summary":pd.DataFrame(rows),"t":ta,"f":fa,"e":ea,"joint":joint}
            except Exception as exc:
                st.error(f"Multi-night processing failed: {exc}")

    if isinstance(st.session_state.multi_result, dict):
        mr = st.session_state.multi_result
        st.dataframe(mr["summary"], use_container_width=True, hide_index=True)
        j = mr["joint"]
        a,b,c = st.columns(3)
        a.metric("Joint period", f"{float(j['period']):.8g} d")
        b.metric("Joint method", str(j.get("search_method","—")))
        c.metric("Combined baseline", f"{np.ptp(mr['t']):.2f} d")
        mf = go.Figure(go.Scatter(x=mr["t"], y=mr["f"], mode="markers", marker=dict(size=4)))
        mf.update_layout(template="plotly_dark", height=350, xaxis_title="Common elapsed-day axis", yaxis_title="Night-normalized flux")
        st.plotly_chart(mf, use_container_width=True)
        ds = mr["summary"].dropna(subset=["Depth_ppm"])
        if len(ds):
            depthfig = go.Figure(go.Scatter(x=ds["File"], y=ds["Depth_ppm"], error_y=dict(type="data",array=ds["Depth_err_ppm"],visible=True), mode="markers"))
            depthfig.update_layout(template="plotly_dark", height=300, yaxis_title="Estimated depth (ppm)")
            st.plotly_chart(depthfig, use_container_width=True)

# === ETC / INSTRUMENT ==========================================================
with tabs[8]:
    st.subheader("Approximate exposure-time / transit detectability calculator")
    st.warning("This fast ETC is a transparent approximation. Use the official instrument ETC / Pandeia/PandExo for proposal-grade JWST calculations.")
    ea, eb, ec, ed = st.columns(4)
    mag = ea.number_input("Target magnitude", value=14.0, step=0.25, key="etc_mag")
    mag_system = eb.selectbox("Magnitude system", ["Vega", "AB"], key="etc_magsys")
    throughput = ec.slider("End-to-end throughput", 0.01, 1.0, 0.25, 0.01, key="etc_throughput")
    in_hours = ed.number_input("In-transit integration (h)", 0.01, 100.0, 5.0, 0.25, key="etc_hours")
    qa,qb,qc,qd = st.columns(4)
    etc_exp = qa.number_input("Exposure time (s)", 0.1, 3600.0, max(1.0, exposure_seconds), 1.0, key="etc_exp")
    dead = qb.number_input("Dead time (s)", 0.0, 600.0, 2.0, 0.5, key="etc_dead")
    aperture_pix = qc.number_input("Photometric aperture (pixels)", 1, 100000, 50, 1, key="etc_ap_pix")
    floor = qd.number_input("Systematic floor (ppm)", 0.0, 100000.0, float(obs_p["systematic_floor_ppm"]), 10.0, key="etc_floor")
    ra,rb,rc = st.columns(3)
    sky = ra.number_input("Sky e−/s/pix", 0.0, 1e8, 1.0 if not obs_p["space"] else 0.05, key="etc_sky")
    dark = rb.number_input("Dark e−/s/pix", 0.0, 1e4, 0.01, key="etc_dark")
    read = rc.number_input("Read noise e−/pix/exposure", 0.0, 1e4, 5.0, key="etc_read")

    transit_depth = engine.rp_rs**2
    etc = ApproximateETC.transit_snr(
        float(mag), mag_system, float(obs_p["diameter_m"]), float(throughput), bandpass, transit_depth,
        float(in_hours), float(etc_exp), float(dead), int(aperture_pix), float(sky), float(dark), float(read), float(floor),
    )
    e1,e2,e3,e4 = st.columns(4)
    e1.metric("Transit SNR", f"{etc['snr']:.2f}")
    e2.metric("Random precision", f"{etc['random_ppm']:.1f} ppm")
    e3.metric("Total precision", f"{etc['total_ppm']:.1f} ppm")
    e4.metric("Exposures", f"{int(etc['n_exp'])}")
    st.caption("Depth SNR assumes an equal-S/N out-of-transit baseline. Scintillation, flat-fielding, slit losses, persistence, saturation, spectral extraction, covariance, and detailed instrument response are not fully modeled by the quick ETC.")

    with st.expander("JWST / PandExo handoff", expanded=False):
        st.write("VORTEX intentionally does not fabricate a PandExo result without Pandeia reference data and a fully specified instrument configuration.")
        pandexo_template = {
            "target": st.session_state.target_display_name,
            "host_temperature_K": host_temp_k,
            "host_radius_Rsun": engine.host_radius_solar,
            "host_mass_Msun": engine.host_mass_solar,
            "planet_radius_Rearth": companion_radius_earth,
            "period_days": period_days,
            "transit_duration_hours": engine.transit_duration_days()*24,
            "note": "Starter metadata only; complete with PandExo/Pandeia instrument and spectrum configuration.",
        }
        st.download_button("Download PandExo starter metadata JSON", json.dumps(pandexo_template, indent=2), "vortex_pandexo_starter.json", "application/json")


    with st.expander("Multi-channel / transmission-spectrum quick-look", expanded=False):
        st.caption(
            "Upload a long-format spectroscopic light-curve table and VORTEX will estimate one transit depth per wavelength channel "
            "using the common ephemeris. This is a quick-look diagnostic, not a replacement for channel-by-channel limb-darkened retrievals."
        )
        spec_file = st.file_uploader("Spectroscopic CSV", type=["csv"], key="spec_file")
        if spec_file is not None:
            try:
                sdf = pd.read_csv(spec_file)
                scols = sdf.columns.tolist()
                sa,sb,sc,sd = st.columns(4)
                stime = sa.selectbox("Time", scols, index=0, key="spec_time")
                sflux = sb.selectbox("Flux", scols, index=default_flux_column_index(scols), key="spec_flux")
                swave = sc.selectbox("Wavelength / channel", scols, index=min(2,len(scols)-1), key="spec_wave")
                serr_opts = ["Estimate from robust scatter"] + scols
                serr = sd.selectbox("Error", serr_opts, index=0, key="spec_err")
                sfmt = guess_time_format_from_column(stime, sdf[stime].to_numpy())
                spec_cad = st.number_input("Frame cadence (min, when applicable)", min_value=0.001, value=float(obs_p["cadence_min"]), step=0.1, key="spec_cad")
                if st.button("Build quick-look transmission spectrum", key="run_spec"):
                    rows_spec=[]
                    for channel, g in sdf.groupby(swave):
                        ti,spec_offset,_,spec_abs = to_internal_time(g[stime].to_numpy(), sfmt, frame_cadence_min=float(spec_cad))
                        if bool(spec_abs) != bool(is_absolute_time):
                            raise ValueError("The spectroscopic table and the active VORTEX dataset must use compatible absolute/relative time conventions for a shared ephemeris.")
                        spec_t0 = recovered_t0 + display_offset - spec_offset if spec_abs else recovered_t0
                        fi = pd.to_numeric(g[sflux],errors="coerce").to_numpy(dtype=float)
                        ei = None if serr=="Estimate from robust scatter" else pd.to_numeric(g[serr],errors="coerce").to_numpy(dtype=float)
                        ti,fi,ei,_ = prepare_lightcurve(ti,fi,ei,clip_positive_spikes=False)
                        dur = max(engine.transit_duration_days(period_days=recovered_p),2*np.median(np.diff(np.sort(ti))))
                        fc,_,ec,_ = gp_detrend_transit_aware(ti,fi,ei,recovered_p,spec_t0,dur,max_gp_points=300)
                        dep = estimate_depth_at_ephemeris(ti,fc,ec,spec_t0,recovered_p,dur)
                        depth=max(dep["depth"],0.0) if np.isfinite(dep["depth"]) else np.nan
                        depth_err=dep["error"]
                        rr=math.sqrt(depth) if np.isfinite(depth) else np.nan
                        rr_err=(depth_err/(2*rr)) if np.isfinite(depth_err) and np.isfinite(rr) and rr>0 else np.nan
                        rows_spec.append({"Channel":channel,"Depth":depth,"Depth_Error":depth_err,"Rp_Rs":rr,"Rp_Rs_Error":rr_err,"N":len(ti)})
                    st.session_state["spec_quicklook"] = pd.DataFrame(rows_spec)
                if isinstance(st.session_state.get("spec_quicklook"),pd.DataFrame):
                    sq=st.session_state["spec_quicklook"]
                    st.dataframe(sq,use_container_width=True,hide_index=True)
                    sx=pd.to_numeric(sq["Channel"],errors="coerce")
                    if np.all(np.isfinite(sx)):
                        spfig=go.Figure(go.Scatter(x=sx,y=sq["Rp_Rs"],error_y=dict(type="data",array=sq["Rp_Rs_Error"],visible=True),mode="markers+lines"))
                        spfig.update_layout(template="plotly_dark",height=320,xaxis_title="Wavelength / channel",yaxis_title="Rp/R* quick-look")
                        st.plotly_chart(spfig,use_container_width=True)
                    st.download_button("Download quick-look spectrum",sq.to_csv(index=False),"vortex_quicklook_spectrum.csv","text/csv")
            except Exception as exc:
                st.error(f"Spectroscopic quick-look failed: {exc}")

# === CATALOG / MISSION =========================================================
with tabs[9]:
    st.subheader("Catalog and external-data integrations")
    left, right = st.columns(2)
    with left:
        st.markdown("#### NASA Exoplanet Archive")
        if isinstance(st.session_state.archive_results, pd.DataFrame):
            st.dataframe(st.session_state.archive_results, use_container_width=True, hide_index=True)
        else:
            st.info("Use the sidebar NASA Archive target source to query PSCompPars.")
        st.caption("Archive values are starting points. VORTEX preserves manual editing because literature solutions and composite values can differ.")
    with right:
        st.markdown("#### TESS / Kepler / K2")
        st.write(f"Lightkurve integration: **{'available' if LIGHTKURVE_AVAILABLE else 'not installed'}**")
        if isinstance(st.session_state.mission_df, pd.DataFrame):
            st.dataframe(st.session_state.mission_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Choose Analyze Real Data → TESS / Kepler / K2 in the sidebar to search and download a light curve.")

    st.markdown("#### Model provenance")
    prov = pd.DataFrame({
        "Component": ["Transit model", "Period search", "Short-baseline search", "GP", "Timing", "Observability", "Limb darkening"],
        "Implementation": ["BATMAN", "Transit Least Squares", "Astropy BoxLeastSquares", "scikit-learn GaussianProcessRegressor", "Astropy Time", "Astropy coordinates / optional astroplan", "Approximate / custom / optional ExoTiC-LD"],
    })
    st.dataframe(prov, use_container_width=True, hide_index=True)

# === DATA / PROJECT / EXPORT ===================================================
with tabs[10]:
    st.subheader("Quality control, reproducibility, projects and export")
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Rows retained", quality_report.get("rows_retained", len(t_obs)))
    q2.metric("Quality-flag removed", flagged_rows_removed)
    q3.metric("Non-finite removed", quality_report.get("nonfinite_removed", 0))
    q4.metric("Duplicates combined", quality_report.get("duplicates_combined", 0))

    export_df = pd.DataFrame({
        "Time_Internal_Days": t_obs,
        f"Time_{display_time_label.replace(' ','_')}": t_obs + display_offset,
        "Raw_Flux": raw_flux,
        "Trend_Model": gp_trend,
        "Detrended_Flux": cleaned_flux,
        "Detrended_Error": cleaned_err,
        "Recovered_Transit_Model": fit_model_current,
        "Residual": cleaned_flux - fit_model_current,
    })
    st.dataframe(export_df.head(200), use_container_width=True, hide_index=True)

    if uploaded_df is not None:
        with st.expander("Manual row-quality editor", expanded=False):
            st.caption("Enter original CSV row numbers to exclude on the next rerun. This complements, rather than overwrites, your file's own quality flag.")
            current_text = ",".join(str(i) for i in st.session_state.get("manual_excluded_rows", []))
            row_text = st.text_input("Rows to exclude (comma-separated, zero-based)", value=current_text, key="manual_rows_text")
            if st.button("Apply manual row exclusions", key="apply_manual_rows"):
                try:
                    rows = [] if not row_text.strip() else sorted(set(int(x.strip()) for x in row_text.split(",") if x.strip()))
                    if any(x < 0 or x >= len(uploaded_df) for x in rows):
                        raise ValueError("One or more row indices are outside the uploaded table.")
                    st.session_state.manual_excluded_rows = rows
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("#### Reproducibility")
    versions = pd.DataFrame({
        "Software": ["VORTEX", "Python", "Streamlit", "NumPy", "Pandas", "Astropy", "BATMAN", "emcee", "transitleastsquares", "scikit-learn", "Lightkurve", "ExoTiC-LD"],
        "Version": [APP_VERSION, f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}", safe_version("streamlit"), safe_version("numpy"), safe_version("pandas"), safe_version("astropy"), safe_version("batman-package"), safe_version("emcee"), safe_version("transitleastsquares"), safe_version("scikit-learn"), safe_version("lightkurve"), safe_version("exotic-ld")],
    })
    st.dataframe(versions, use_container_width=True, hide_index=True)

    methods = analysis_methods_text(engine, obs_name, search_result, gp_kernel, detection_mode, display_time_label, len(t_obs))
    project_config = {
        "vortex_version": APP_VERSION,
        "target_name": st.session_state.target_display_name,
        "widget_state": {
            "host_mass_jup": host_mass_jup, "host_radius_jup": host_radius_jup,
            "companion_radius_earth": companion_radius_earth, "companion_mass_earth": companion_mass_earth,
            "period_days": period_days, "inclination_deg": inclination_deg, "eccentricity": eccentricity,
            "omega_deg": omega_deg, "host_temp_k": host_temp_k, "companion_temp_k": companion_temp_k,
            "target_logg": target_logg, "target_metallicity": target_metallicity,
            "target_ra_deg": target_ra_deg, "target_dec_deg": target_dec_deg,
            "observatory_name": obs_name, "bandpass": bandpass, "workflow_mode": workflow,
        },
        "search_result": {k: (v.tolist() if isinstance(v,np.ndarray) else v) for k,v in search_result.items() if k not in {"model_lightcurve_time","model_lightcurve_model","folded_phase","folded_y","folded_dy","model_folded_phase","model_folded_model"}},
        "quality_report": quality_report,
        "manual_excluded_rows": st.session_state.get("manual_excluded_rows", []),
    }

    project_upload = st.file_uploader("Load VORTEX project JSON", type=["json"], key="project_upload")
    if project_upload is not None and st.button("Apply project settings", key="apply_project"):
        try:
            cfg = json.load(project_upload)
            apply_project_state(cfg)
            st.rerun()
        except Exception as exc:
            st.error(f"Project file could not be applied: {exc}")

    e1,e2,e3,e4 = st.columns(4)
    e1.download_button("Light curve CSV", export_df.to_csv(index=False), "vortex_lightcurve.csv", "text/csv", use_container_width=True)
    e2.download_button("Light curve FITS", dataframe_to_fits_bytes(export_df), "vortex_lightcurve.fits", "application/fits", use_container_width=True)
    e3.download_button("Project JSON", json.dumps(project_config, indent=2, default=str), "vortex_project.json", "application/json", use_container_width=True)
    e4.download_button("Methods TXT", methods, "vortex_methods.txt", "text/plain", use_container_width=True)

    # Compact analysis bundle.
    extra_tables = {"candidates.csv": candidate_df}
    if isinstance(st.session_state.injrec_result, pd.DataFrame): extra_tables["injection_recovery.csv"] = st.session_state.injrec_result
    if isinstance(st.session_state.planner_result, pd.DataFrame): extra_tables["observing_plan.csv"] = st.session_state.planner_result
    if isinstance(st.session_state.multi_result, dict): extra_tables["multi_night_summary.csv"] = st.session_state.multi_result["summary"]
    try:
        periodogram_df = pd.DataFrame({"Period_Days": period_grid, "Power": power_grid}) if len(period_grid) == len(power_grid) else pd.DataFrame()
        posterior_export = None
        if isinstance(st.session_state.mcmc_pro, dict) and np.asarray(st.session_state.mcmc_pro.get("flat", [])).size:
            posterior_export = pd.DataFrame(st.session_state.mcmc_pro["flat"], columns=st.session_state.mcmc_pro["names"])
        bundle = build_analysis_zip(project_config, export_df, periodogram_df, candidate_df, methods, posterior_export, extra_tables)
        st.download_button("📦 Download reproducible analysis bundle", bundle, "vortex_analysis_bundle.zip", "application/zip", use_container_width=True)
    except Exception as exc:
        st.caption(f"Bundle export unavailable: {exc}")

# Footer / developer credit — intentionally retained.
st.divider()
st.markdown(
    "<p style='text-align:center;color:#7890aa;font-size:.88rem;'>"
    f"VORTEX Professional v{APP_VERSION} | Designed &amp; Developed by <b>Amanpreet Singh</b><br>"
    "Astrophysics &amp; Transit Modeling<br>"
    "Scientific software prototype: inspect assumptions, convergence, and instrument-specific limitations before publication use."
    "</p>",
    unsafe_allow_html=True,
)
