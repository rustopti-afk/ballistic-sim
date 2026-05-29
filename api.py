"""
Ballistic Trajectory Simulation API
====================================

Flask HTTP server exposing the trajectory simulator.

Endpoints
---------
GET  /health      -> {"status": "ok"}
POST /calculate   -> Run simulate() with JSON body:
                     {lat1, lon1, lat2, lon2, mass, diameter, cd}

CORS is enabled for all origins. Server binds 0.0.0.0:5000.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from physics.trajectory import simulate

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ballistic-api")


REQUIRED_FIELDS = ("lat1", "lon1", "lat2", "lon2", "mass", "diameter", "cd")


def _coerce_float(payload: dict, key: str) -> float:
    """Return payload[key] as float or raise ValueError with a clear message."""
    if key not in payload:
        raise ValueError(f"Missing required field: '{key}'")
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"Field '{key}' must be a number, got {payload[key]!r}")


@app.route("/")
def index() -> Any:
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/calculate", methods=["POST", "OPTIONS"])
def calculate() -> Any:
    """
    Execute a ballistic trajectory simulation.

    Request JSON body
    -----------------
    lat1, lon1 : launch coordinates (deg)
    lat2, lon2 : target coordinates (deg)
    mass       : projectile mass (kg)
    diameter   : projectile diameter (m)
    cd         : drag coefficient (dimensionless)

    Response JSON
    -------------
    The dictionary returned by simulate().
    """
    if request.method == "OPTIONS":
        # CORS preflight short-circuit.
        return ("", 204)

    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({"success": False, "error": "Body must be JSON object"}), 400

    try:
        lat1 = _coerce_float(payload, "lat1")
        lon1 = _coerce_float(payload, "lon1")
        lat2 = _coerce_float(payload, "lat2")
        lon2 = _coerce_float(payload, "lon2")
        mass = _coerce_float(payload, "mass")
        diameter = _coerce_float(payload, "diameter")
        cd = _coerce_float(payload, "cd")
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    try:
        result = simulate(lat1, lon1, lat2, lon2, mass, diameter, cd)
        return jsonify(result)
    except ValueError as exc:
        # Domain errors raised by simulate() input validation.
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log.error("Simulation crashed: %s\n%s", exc, traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Internal simulation error: {exc}",
        }), 500


@app.errorhandler(404)
def not_found(_err: Any) -> Any:
    return jsonify({"success": False, "error": "Not Found"}), 404


@app.errorhandler(405)
def method_not_allowed(_err: Any) -> Any:
    return jsonify({"success": False, "error": "Method Not Allowed"}), 405


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
