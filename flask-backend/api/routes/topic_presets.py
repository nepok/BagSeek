"""Topic preset routes."""
from flask import Blueprint, jsonify, request
from ..utils.topic_presets import load_topic_presets, save_topic_presets

topic_presets_bp = Blueprint('topic_presets', __name__)


@topic_presets_bp.route("/api/topic-presets", methods=["GET"])
def api_load_topic_presets():
    """Load all topic presets."""
    return jsonify(load_topic_presets())


@topic_presets_bp.route("/api/save-topic-preset", methods=["POST"])
def api_save_topic_preset():
    """Save a topic preset. Body: { name: string, topics: string[] }"""
    data = request.json
    name = data.get("name")
    topics = data.get("topics")

    if not name or not isinstance(name, str):
        return jsonify({"error": "Name is required"}), 400
    if not isinstance(topics, list):
        return jsonify({"error": "Topics must be a list"}), 400

    presets = load_topic_presets()
    presets[name] = topics
    save_topic_presets(presets)

    return jsonify({"message": f"Topic preset '{name}' saved successfully"})


@topic_presets_bp.route("/api/delete-topic-preset", methods=["POST"])
def api_delete_topic_preset():
    """Delete a topic preset. Body: { name: string }"""
    data = request.json
    name = data.get("name")
    presets = load_topic_presets()

    if name in presets:
        del presets[name]
        save_topic_presets(presets)
        return jsonify({"message": f"Topic preset '{name}' deleted successfully"})

    return jsonify({"error": "Preset not found"}), 404
