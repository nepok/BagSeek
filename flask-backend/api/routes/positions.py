"""Position routes."""
from flask import Blueprint, jsonify
from ..utils.rosbag import _load_positional_lookup

positions_bp = Blueprint('positions', __name__)


@positions_bp.route('/api/positions/rosbags', methods=['GET'])
def get_positions_rosbags():
    """
    Return the list of rosbag names available in the positional lookup table.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_names = sorted(lookup.keys())
        return jsonify({"rosbags": rosbag_names}), 200
    except FileNotFoundError:
        return jsonify({"rosbags": []}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@positions_bp.route('/api/positions/rosbags/<path:rosbag_name>', methods=['GET'])
def get_positional_rosbag_entries(rosbag_name: str):
    """
    Return the positional lookup entries for a specific rosbag.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        points = []
        for lat_lon, location_data in rosbag_data.items():
            try:
                lat_str, lon_str = lat_lon.split(',')
                count = int(location_data["total"])
                
                points.append({
                    "lat": float(lat_str),
                    "lon": float(lon_str),
                    "count": count
                })
            except (ValueError, TypeError, KeyError):
                continue

        points.sort(key=lambda item: item["count"], reverse=True)

        return jsonify({
            "rosbag": rosbag_name,
            "points": points
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@positions_bp.route('/api/positions/rosbags/<path:rosbag_name>/mcaps', methods=['GET'])
def get_positional_rosbag_mcaps(rosbag_name: str):
    """
    Return positional lookup entries for a specific rosbag grouped by location with per-mcap breakdown.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        points = []
        for lat_lon, location_data in rosbag_data.items():
            try:
                lat_str, lon_str = lat_lon.split(',')
                mcaps = location_data.get("mcaps", {})
                
                # Group by location, include all mcaps at this location
                if mcaps:
                    points.append({
                        "lat": float(lat_str),
                        "lon": float(lon_str),
                        "mcaps": {mcap_id: int(count) for mcap_id, count in mcaps.items()}
                    })
            except (ValueError, TypeError, KeyError):
                continue

        return jsonify({
            "rosbag": rosbag_name,
            "points": points
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@positions_bp.route('/api/positions/rosbags/<path:rosbag_name>/mcap-list', methods=['GET'])
def get_positional_rosbag_mcap_list(rosbag_name: str):
    """
    Return positional lookup entries for a specific rosbag grouped by location with per-mcap breakdown.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        mcap_counts: dict[str, int] = {}
        for location_data in rosbag_data.values():
            mcaps = location_data.get("mcaps", {})
            for mcap_id, count in mcaps.items():
                mcap_counts[mcap_id] = mcap_counts.get(mcap_id, 0) + int(count)

        # Convert to list of dicts and sort by mcap_id (numeric if possible, otherwise alphabetical)
        def sort_key(item):
            mcap_id = item[0]
            # Try to parse as integer for numeric sorting, otherwise use string
            try:
                return (0, int(mcap_id))  # Numeric IDs first
            except ValueError:
                return (1, mcap_id)  # Non-numeric IDs after
        
        mcap_list = [
            {"id": mcap_id, "totalCount": count}
            for mcap_id, count in sorted(mcap_counts.items(), key=sort_key)
        ]

        return jsonify({
            "rosbag": rosbag_name,
            "mcaps": mcap_list
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@positions_bp.route('/api/positions/all', methods=['GET'])
def get_positions_all():
    """
    Return aggregated positional lookup entries across all rosbags.
    """
    try:
        lookup = _load_positional_lookup()
        
        # Aggregate across all rosbags
        aggregated: dict[str, dict[str, int | float]] = {}
        for rosbag_data in lookup.values():
            for lat_lon, location_data in rosbag_data.items():
                try:
                    lat_str, lon_str = lat_lon.split(',')
                    count = int(location_data["total"])
                    
                    key = f"{float(lat_str):.6f},{float(lon_str):.6f}"
                    if key not in aggregated:
                        aggregated[key] = {
                            "lat": float(lat_str),
                            "lon": float(lon_str),
                            "count": count,
                        }
                    else:
                        aggregated[key]["count"] = int(aggregated[key]["count"]) + count  # type: ignore[index]
                except (ValueError, TypeError, KeyError):
                    continue

        points = sorted(
            (
                {
                    "lat": value["lat"],
                    "lon": value["lon"],
                    "count": int(value["count"]),
                }
                for value in aggregated.values()
            ),
            key=lambda item: item["count"],
            reverse=True,
        )

        return jsonify({"points": points}), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
