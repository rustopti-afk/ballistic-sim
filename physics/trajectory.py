"""
Ballistic Trajectory Simulation Engine
=======================================

Production-ready 6-DOF (translational) ballistic trajectory simulator using an
ECEF (Earth-Centered Earth-Fixed) coordinate frame.

Forces modeled:
    * Newtonian gravity (inverse-square with altitude)
    * Aerodynamic drag (ISA atmosphere, velocity relative to rotating air mass)
    * Coriolis force (ECEF rotating frame)
    * Centrifugal force (ECEF rotating frame)

Integration: scipy.integrate.solve_ivp (RK45) with an impact event terminator.

Public API:
    simulate(lat1_deg, lon1_deg, lat2_deg, lon2_deg, mass_kg, diameter_m, cd)
    latlon_to_ecef(lat_deg, lon_deg, alt_m=0)
    ecef_to_latlon(r)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R_EARTH: float = 6_371_000.0           # Spherical Earth radius [m]
G0: float = 9.80665                    # Standard gravity at surface [m/s^2]
OMEGA_EARTH: float = 7.2921e-5         # Earth sidereal rotation rate [rad/s]
R_SPECIFIC_AIR: float = 287.05         # Specific gas constant for dry air [J/(kg*K)]

# Earth angular velocity vector in ECEF (z-axis aligned with rotation axis).
OMEGA_VEC: np.ndarray = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)


# ---------------------------------------------------------------------------
# Atmosphere (International Standard Atmosphere, simplified)
# ---------------------------------------------------------------------------

def isa_density(h: float) -> float:
    """
    Return air density [kg/m^3] at geometric altitude h [m] using a piecewise
    ISA-style atmosphere. Negative altitudes are clamped to surface.

    Layers:
        0    <= h < 11000  : troposphere, linear lapse 6.5 K/km
        11000 <= h < 20000 : lower stratosphere, isothermal
        20000 <= h < 32000 : mid stratosphere, inverse lapse 1 K/km
        h    >= 32000      : exponential decay (continuous with layer 3)
    """
    if h < 0.0:
        h = 0.0

    if h < 11_000.0:
        T = 288.15 - 0.0065 * h
        P = 101_325.0 * (T / 288.15) ** 5.2561
    elif h < 20_000.0:
        T = 216.65
        P = 22_632.1 * math.exp(-0.0001577 * (h - 11_000.0))
    elif h < 32_000.0:
        T = 216.65 + 0.001 * (h - 20_000.0)
        P = 5_474.89 * (T / 216.65) ** -34.1632
    else:
        # Above 32 km: continue with an exponential decay anchored at 32 km.
        # Scale height H ~ R*T/g0 with T_ref ~ 228.65 K -> ~6700 m.
        T_ref = 228.65
        P_ref = 868.0187  # approx pressure at 32 km from layer 3 formula
        H_scale = R_SPECIFIC_AIR * T_ref / G0
        P = P_ref * math.exp(-(h - 32_000.0) / H_scale)
        T = T_ref
        if h > 120_000.0:
            # Effectively vacuum above ~120 km for ballistic purposes.
            return 0.0

    rho = P / (R_SPECIFIC_AIR * T)
    return max(rho, 0.0)


# ---------------------------------------------------------------------------
# Coordinate conversions (spherical Earth model)
# ---------------------------------------------------------------------------

def latlon_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """
    Convert geodetic (lat, lon, alt) to ECEF Cartesian coordinates [m].

    Uses a spherical Earth (radius R_EARTH). Lat/lon are in degrees.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = R_EARTH + alt_m
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def ecef_to_latlon(r: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF Cartesian coordinates [m] to (lat_deg, lon_deg, alt_m).

    Uses spherical Earth assumption.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    radius = math.sqrt(x * x + y * y + z * z)
    if radius == 0.0:
        return 0.0, 0.0, -R_EARTH
    lat = math.degrees(math.asin(z / radius))
    lon = math.degrees(math.atan2(y, x))
    alt = radius - R_EARTH
    return lat, lon, alt


# ---------------------------------------------------------------------------
# Local frame helpers
# ---------------------------------------------------------------------------

def _enu_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (east, north, up) unit vectors in ECEF for the given lat/lon.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return east, north, up


def _great_circle_bearing(lat1_deg: float, lon1_deg: float,
                          lat2_deg: float, lon2_deg: float) -> float:
    """
    Initial great-circle bearing from point 1 to point 2 [radians, 0 = north,
    clockwise].
    """
    phi1 = math.radians(lat1_deg)
    phi2 = math.radians(lat2_deg)
    dlon = math.radians(lon2_deg - lon1_deg)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.atan2(x, y)


def _great_circle_distance(lat1_deg: float, lon1_deg: float,
                           lat2_deg: float, lon2_deg: float) -> float:
    """Great-circle surface distance [m] between two lat/lon points."""
    phi1 = math.radians(lat1_deg)
    phi2 = math.radians(lat2_deg)
    dphi = phi2 - phi1
    dlon = math.radians(lon2_deg - lon1_deg)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R_EARTH * c


# ---------------------------------------------------------------------------
# Equations of motion in ECEF
# ---------------------------------------------------------------------------

def _equations_of_motion(t: float, state: np.ndarray,
                         mass_kg: float, area_m2: float, cd: float) -> np.ndarray:
    """
    State derivative for [x, y, z, vx, vy, vz] in ECEF.

    Forces:
        * Gravity: g(h) = G0 * (R / (R + h))^2 toward Earth center
        * Drag:    -0.5 * rho(h) * Cd * A * |v_rel| * v_rel
                   where v_rel = v - omega x r (velocity relative to rotating
                   atmosphere co-rotating with Earth)
        * Coriolis:    -2 * (omega x v)
        * Centrifugal: -(omega x (omega x r))
    """
    r = state[0:3]
    v = state[3:6]

    radius = float(np.linalg.norm(r))
    if radius < 1.0:
        # Numerical safety: avoid divide-by-zero at center.
        return np.zeros(6)

    altitude = radius - R_EARTH

    # Gravity (inverse-square, toward center)
    g_mag = G0 * (R_EARTH / (R_EARTH + max(altitude, 0.0))) ** 2
    a_gravity = -g_mag * (r / radius)

    # Atmosphere co-rotates with Earth: v_rel = v_inertial_relative_to_air
    # In ECEF, the air is (approximately) stationary, so v_rel = v.
    # The Coriolis/centrifugal terms below account for the rotating frame.
    v_rel = v
    v_rel_mag = float(np.linalg.norm(v_rel))

    # Drag
    if v_rel_mag > 0.0 and altitude < 150_000.0:
        rho = isa_density(altitude)
        drag_force_mag = 0.5 * rho * cd * area_m2 * v_rel_mag * v_rel_mag
        a_drag = -(drag_force_mag / mass_kg) * (v_rel / v_rel_mag)
    else:
        a_drag = np.zeros(3)

    # Coriolis: -2 * omega x v
    a_coriolis = -2.0 * np.cross(OMEGA_VEC, v)

    # Centrifugal: -omega x (omega x r)
    a_centrifugal = -np.cross(OMEGA_VEC, np.cross(OMEGA_VEC, r))

    a_total = a_gravity + a_drag + a_coriolis + a_centrifugal

    return np.array([v[0], v[1], v[2], a_total[0], a_total[1], a_total[2]])


def _impact_event(t: float, state: np.ndarray, *args) -> float:
    """Event: |r| - R_EARTH = 0 -> impact with ground."""
    return float(np.linalg.norm(state[0:3]) - R_EARTH)


_impact_event.terminal = True   # type: ignore[attr-defined]
_impact_event.direction = -1.0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Single trajectory integration
# ---------------------------------------------------------------------------

def _initial_velocity_vector(lat_deg: float, lon_deg: float,
                             speed: float, elevation_rad: float,
                             azimuth_rad: float) -> np.ndarray:
    """
    Build an initial inertial velocity vector in ECEF from a launch point and
    local (azimuth, elevation, speed) tuple.

    Azimuth is measured clockwise from north (0 = N, pi/2 = E).
    Elevation is measured up from local horizontal.

    Returns the ECEF velocity vector accounting for the launch point's
    co-rotation with Earth (so the integrator's ECEF velocity matches what an
    observer on the ground would call "muzzle velocity").
    """
    east, north, up = _enu_basis(lat_deg, lon_deg)
    horiz = math.cos(elevation_rad)
    vert = math.sin(elevation_rad)
    direction = (
        horiz * math.sin(azimuth_rad) * east +
        horiz * math.cos(azimuth_rad) * north +
        vert * up
    )
    return speed * direction


def _integrate_trajectory(r0: np.ndarray, v0: np.ndarray,
                          mass_kg: float, area_m2: float, cd: float,
                          t_max: float = 7200.0, max_step: float = 30.0,
                          dense: bool = False):
    state0 = np.concatenate([r0, v0])
    sol = solve_ivp(
        fun=lambda t, y: _equations_of_motion(t, y, mass_kg, area_m2, cd),
        t_span=(0.0, t_max),
        y0=state0,
        method="RK45",
        events=_impact_event,
        max_step=max_step,
        rtol=1e-4,
        atol=1.0,
        dense_output=dense,
    )
    return sol


# ---------------------------------------------------------------------------
# Launch parameter search (minimum-energy bisection over elevation angle)
# ---------------------------------------------------------------------------

def _impact_point_for_angle(lat1: float, lon1: float,
                            speed: float, elevation_rad: float,
                            azimuth_rad: float,
                            mass_kg: float, area_m2: float, cd: float
                            ) -> Tuple[float, float, "object"]:
    """
    Fire a projectile and return (impact_lat, impact_lon, solution_object).
    If no impact within t_max, returns the final-state lat/lon anyway.
    """
    r0 = latlon_to_ecef(lat1, lon1, 0.0)
    v0 = _initial_velocity_vector(lat1, lon1, speed, elevation_rad, azimuth_rad)
    sol = _integrate_trajectory(r0, v0, mass_kg, area_m2, cd, dense=False)
    final_r = sol.y[0:3, -1]
    lat, lon, _alt = ecef_to_latlon(final_r)
    return lat, lon, sol


def _range_for_angle(lat1: float, lon1: float, lat2: float, lon2: float,
                     speed: float, elevation_rad: float, azimuth_rad: float,
                     mass_kg: float, area_m2: float, cd: float) -> Tuple[float, "object"]:
    """
    Return (great-circle distance from start to impact [m], solution).
    """
    imp_lat, imp_lon, sol = _impact_point_for_angle(
        lat1, lon1, speed, elevation_rad, azimuth_rad,
        mass_kg, area_m2, cd
    )
    distance = _great_circle_distance(lat1, lon1, imp_lat, imp_lon)
    return distance, sol


def _find_required_speed(lat1: float, lon1: float, lat2: float, lon2: float,
                         elevation_rad: float, azimuth_rad: float,
                         mass_kg: float, area_m2: float, cd: float,
                         target_distance: float) -> Tuple[float, "object"]:
    # Physics estimate for initial speed bracket (vacuum, 45 deg)
    d = target_distance
    v_est = math.sqrt(G0 * d)  # order-of-magnitude estimate
    lo = max(100.0, v_est * 0.3)
    hi = min(12_000.0, v_est * 4.0)

    best_sol = None
    for _ in range(20):  # 20 замість 60
        mid = 0.5 * (lo + hi)
        dist, sol = _range_for_angle(
            lat1, lon1, lat2, lon2,
            mid, elevation_rad, azimuth_rad,
            mass_kg, area_m2, cd
        )
        best_sol = sol
        if dist < target_distance:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 10.0:
            break
    return 0.5 * (lo + hi), best_sol


def _optimize_launch(lat1: float, lon1: float, lat2: float, lon2: float,
                     mass_kg: float, area_m2: float, cd: float):
    azimuth = _great_circle_bearing(lat1, lon1, lat2, lon2)
    target_distance = _great_circle_distance(lat1, lon1, lat2, lon2)

    # 4 кути замість 13 — достатньо для знаходження мінімальної енергії
    elevations_deg = [30.0, 45.0, 60.0, 75.0]
    best = None

    for el_deg in elevations_deg:
        el_rad = math.radians(el_deg)
        speed, sol = _find_required_speed(
            lat1, lon1, lat2, lon2,
            el_rad, azimuth, mass_kg, area_m2, cd, target_distance
        )
        final_r = sol.y[0:3, -1]
        imp_lat, imp_lon, _ = ecef_to_latlon(final_r)
        miss = abs(_great_circle_distance(imp_lat, imp_lon, lat2, lon2))

        if best is None or speed < best[0]:
            best = (speed, el_rad, sol, miss)

    if best is None:
        raise RuntimeError("Launch optimization failed.")

    speed, el_rad, sol, miss = best
    return speed, el_rad, azimuth, sol, miss


# ---------------------------------------------------------------------------
# Public simulate() entry point
# ---------------------------------------------------------------------------

def simulate(lat1_deg: float, lon1_deg: float,
             lat2_deg: float, lon2_deg: float,
             mass_kg: float, diameter_m: float, cd: float) -> dict:
    """
    Run a full ballistic simulation from (lat1, lon1) to (lat2, lon2).

    Parameters
    ----------
    lat1_deg, lon1_deg : float
        Launch site geodetic coordinates [deg].
    lat2_deg, lon2_deg : float
        Target geodetic coordinates [deg].
    mass_kg : float
        Projectile mass [kg]. Must be > 0.
    diameter_m : float
        Projectile diameter [m]. Must be > 0. Cross-section area = pi*d^2/4.
    cd : float
        Dimensionless drag coefficient. Typical 0.1 - 1.5.

    Returns
    -------
    dict with keys:
        trajectory          list[[lat_deg, lon_deg, alt_km], ...]
        max_altitude_km     float
        flight_time_min     float
        initial_speed_kms   float
        distance_km         float
        success             bool
    """
    # ---- Input validation -------------------------------------------------
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive")
    if diameter_m <= 0.0:
        raise ValueError("diameter_m must be positive")
    if cd < 0.0:
        raise ValueError("cd must be non-negative")
    for name, val in (("lat1", lat1_deg), ("lat2", lat2_deg)):
        if not -90.0 <= val <= 90.0:
            raise ValueError(f"{name} out of range [-90, 90]")
    for name, val in (("lon1", lon1_deg), ("lon2", lon2_deg)):
        if not -180.0 <= val <= 180.0:
            raise ValueError(f"{name} out of range [-180, 180]")

    area_m2 = math.pi * (diameter_m * 0.5) ** 2
    distance_km = _great_circle_distance(lat1_deg, lon1_deg,
                                         lat2_deg, lon2_deg) / 1000.0

    try:
        speed, el_rad, az_rad, _quick_sol, miss = _optimize_launch(
            lat1_deg, lon1_deg, lat2_deg, lon2_deg,
            mass_kg, area_m2, cd
        )
    except Exception as exc:
        return {
            "trajectory": [],
            "max_altitude_km": 0.0,
            "flight_time_min": 0.0,
            "initial_speed_kms": 0.0,
            "distance_km": distance_km,
            "success": False,
            "error": str(exc),
        }

    # Re-integrate with dense_output for smooth resampling
    r0 = latlon_to_ecef(lat1_deg, lon1_deg, 0.0)
    v0 = _initial_velocity_vector(lat1_deg, lon1_deg, speed, el_rad, az_rad)
    sol = _integrate_trajectory(r0, v0, mass_kg, area_m2, cd, dense=True)

    # Resample 80 evenly-spaced points along the trajectory
    trajectory: list = []
    max_alt = 0.0
    t_end = float(sol.t[-1])
    t_samples = np.linspace(0.0, t_end, 80)
    for t in t_samples:
        state = sol.sol(t)
        r = state[0:3]
        lat, lon, alt = ecef_to_latlon(r)
        alt_km = alt / 1000.0
        if alt_km > max_alt:
            max_alt = alt_km
        trajectory.append([lat, lon, alt_km])

    flight_time_min = t_end / 60.0
    # Accept up to 3% miss or 20 km, whichever is larger
    tolerance_m = max(20_000.0, 0.03 * distance_km * 1000.0)
    success = bool(sol.status >= 0 and miss < tolerance_m and len(trajectory) >= 2)

    return {
        "trajectory": trajectory,
        "max_altitude_km": float(max_alt),
        "flight_time_min": float(flight_time_min),
        "initial_speed_kms": float(speed) / 1000.0,
        "distance_km": float(distance_km),
        "success": success,
    }


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Kyiv -> Berlin
    result = simulate(
        lat1_deg=50.4501, lon1_deg=30.5234,
        lat2_deg=52.5200, lon2_deg=13.4050,
        mass_kg=500.0, diameter_m=0.5, cd=0.3,
    )
    print(f"Success:           {result['success']}")
    print(f"Distance:          {result['distance_km']:.1f} km")
    print(f"Initial speed:     {result['initial_speed_kms']:.3f} km/s")
    print(f"Max altitude:      {result['max_altitude_km']:.1f} km")
    print(f"Flight time:       {result['flight_time_min']:.2f} min")
    print(f"Trajectory points: {len(result['trajectory'])}")
