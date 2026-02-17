"""
Audit script for completion.json files.

Checks every completion entry against the actual files on disk.
Reports mismatches (ghost entries, missing files) and optionally repairs them.

Usage:
    python preprocessing/audit_completion.py          # Report only
    python preprocessing/audit_completion.py --fix    # Report and remove broken entries
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import Config


# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def check_output_files(
    base_dir: Path, output_files: list
) -> Tuple[List[str], List[str]]:
    """Check if output_files exist on disk. Returns (existing, missing)."""
    existing = []
    missing = []
    for f in output_files:
        p = Path(f)
        full = p if p.is_absolute() else base_dir / p
        if full.exists():
            existing.append(f)
        else:
            missing.append(f)
    return existing, missing


def audit_rosbags_flat(
    processor_name: str,
    processor_data: Dict[str, Any],
    base_dir: Path,
) -> List[Dict]:
    """Audit: rosbags[name] -> {status, output_files}."""
    issues = []
    rosbags = processor_data.get("rosbags", {})
    for rosbag_name, entry in rosbags.items():
        if not isinstance(entry, dict):
            issues.append({
                "processor": processor_name, "rosbag": rosbag_name,
                "issue": "missing_files",
                "detail": f"Invalid entry: expected dict, got {type(entry).__name__}",
            })
            continue

        if entry.get("status") != "completed":
            continue

        output_files = entry.get("output_files", [])
        if not output_files:
            # Completed with no output_files is legitimate (e.g. no-image rosbag parts)
            continue

        _, missing = check_output_files(base_dir, output_files)
        if missing:
            issues.append({
                "processor": processor_name, "rosbag": rosbag_name,
                "issue": "missing_files",
                "detail": f"{len(missing)} file(s) missing: {missing}",
            })

    return issues


def audit_rosbags_with_mcaps(
    processor_name: str,
    processor_data: Dict[str, Any],
    base_dir: Path,
) -> List[Dict]:
    """Audit: rosbags[name] -> {status, mcaps: {name: {...}}}."""
    issues = []
    rosbags = processor_data.get("rosbags", {})
    for rosbag_name, rosbag_entry in rosbags.items():
        if not isinstance(rosbag_entry, dict):
            continue

        rosbag_status = rosbag_entry.get("status")
        mcaps = rosbag_entry.get("mcaps", {})

        # Check rosbag-level output_files (e.g. summary.json)
        rosbag_output = rosbag_entry.get("output_files", [])
        if rosbag_status == "completed" and rosbag_output:
            _, missing = check_output_files(base_dir, rosbag_output)
            if missing:
                issues.append({
                    "processor": processor_name, "rosbag": rosbag_name,
                    "issue": "missing_files",
                    "detail": f"Rosbag output missing: {missing}",
                })

        # Check each MCAP's output_files
        for mcap_name, mcap_entry in mcaps.items():
            if not isinstance(mcap_entry, dict):
                continue
            if mcap_entry.get("status") != "completed":
                continue

            mcap_output = mcap_entry.get("output_files", [])
            if not mcap_output:
                continue

            _, missing = check_output_files(base_dir, mcap_output)
            if missing:
                issues.append({
                    "processor": processor_name, "rosbag": rosbag_name,
                    "mcap": mcap_name,
                    "issue": "missing_files",
                    "detail": f"MCAP output missing: {[Path(m).name for m in missing]}",
                })

    return issues


def audit_models_with_rosbags_mcaps(
    processor_name: str,
    processor_data: Dict[str, Any],
    base_dir: Path,
) -> List[Dict]:
    """Audit: models[m] -> rosbags[r] -> mcaps[c]."""
    issues = []
    models = processor_data.get("models", {})
    for model_name, model_entry in models.items():
        if not isinstance(model_entry, dict):
            continue
        rosbags = model_entry.get("rosbags", {})
        for rosbag_name, rosbag_entry in rosbags.items():
            if not isinstance(rosbag_entry, dict):
                continue

            rosbag_output = rosbag_entry.get("output_files", [])
            if rosbag_entry.get("status") == "completed" and rosbag_output:
                _, missing = check_output_files(base_dir, rosbag_output)
                if missing:
                    issues.append({
                        "processor": processor_name,
                        "model": model_name, "rosbag": rosbag_name,
                        "issue": "missing_files",
                        "detail": f"{len(missing)} output file(s) missing",
                    })

    return issues


def audit_models_with_rosbags_topics(
    processor_name: str,
    processor_data: Dict[str, Any],
    base_dir: Path,
) -> List[Dict]:
    """Audit: models[m] -> rosbags[r] -> topics[t]."""
    issues = []
    models = processor_data.get("models", {})
    for model_name, model_entry in models.items():
        if not isinstance(model_entry, dict):
            continue
        rosbags = model_entry.get("rosbags", {})
        for rosbag_name, rosbag_entry in rosbags.items():
            if not isinstance(rosbag_entry, dict):
                continue

            topics = rosbag_entry.get("topics", {})
            for topic_name, topic_entry in topics.items():
                if not isinstance(topic_entry, dict):
                    continue
                if topic_entry.get("status") != "completed":
                    continue

                topic_output = topic_entry.get("output_files", [])
                if not topic_output:
                    continue

                _, missing = check_output_files(base_dir, topic_output)
                if missing:
                    issues.append({
                        "processor": processor_name,
                        "model": model_name, "rosbag": rosbag_name,
                        "topic": topic_name,
                        "issue": "missing_files",
                        "detail": f"{len(missing)} file(s) missing: {[Path(m).name for m in missing]}",
                    })

    return issues


def fix_issues(
    completion_path: Path,
    full_data: Dict[str, Any],
    issues: List[Dict],
) -> int:
    """Remove broken entries from completion data and save. Returns count removed."""
    removed = 0

    for issue in issues:
        processor_name = issue["processor"]
        processor_data = full_data.get(processor_name, {})

        if "mcap" in issue:
            # MCAP with missing output — remove the MCAP entry
            rosbags = processor_data.get("rosbags", {})
            rosbag_entry = rosbags.get(issue["rosbag"], {})
            mcaps = rosbag_entry.get("mcaps", {})
            if issue["mcap"] in mcaps:
                del mcaps[issue["mcap"]]
                removed += 1
                # Invalidate rosbag-level completion (no longer fully complete)
                for key in ("status", "completed_at", "output_files"):
                    rosbag_entry.pop(key, None)

        elif "topic" in issue:
            # Topic with missing output — remove the topic entry
            models = processor_data.get("models", {})
            model_entry = models.get(issue["model"], {})
            rosbags = model_entry.get("rosbags", {})
            rosbag_entry = rosbags.get(issue["rosbag"], {})
            topics = rosbag_entry.get("topics", {})
            if issue["topic"] in topics:
                del topics[issue["topic"]]
                removed += 1
                rosbag_entry.pop("status", None)

        elif "model" in issue:
            # Model+rosbag level missing output — clear rosbag status
            models = processor_data.get("models", {})
            model_entry = models.get(issue["model"], {})
            rosbags = model_entry.get("rosbags", {})
            rosbag_entry = rosbags.get(issue["rosbag"], {})
            if isinstance(rosbag_entry, dict) and "status" in rosbag_entry:
                for key in ("status", "completed_at", "output_files"):
                    rosbag_entry.pop(key, None)
                removed += 1

        else:
            # Rosbag-level missing output — clear rosbag status
            rosbags = processor_data.get("rosbags", {})
            rosbag_entry = rosbags.get(issue["rosbag"], {})
            if isinstance(rosbag_entry, dict) and "status" in rosbag_entry:
                for key in ("status", "completed_at", "output_files"):
                    rosbag_entry.pop(key, None)
                removed += 1

    if removed > 0:
        with open(completion_path, "w") as f:
            json.dump(full_data, f, indent=2)

    return removed


def audit_completion_file(
    completion_path: Path,
    processors: List[Dict[str, Any]],
    do_fix: bool = False,
) -> Tuple[List[Dict], int]:
    """Audit a single completion.json file for all processors it contains."""
    if not completion_path.exists():
        return [], 0

    with open(completion_path) as f:
        full_data = json.load(f)

    all_issues = []

    for proc in processors:
        name = proc["name"]
        audit_type = proc["type"]
        base_dir = proc["base_dir"]

        processor_data = full_data.get(name, {})
        if not processor_data:
            continue

        if audit_type == "rosbags_flat":
            issues = audit_rosbags_flat(name, processor_data, base_dir)
        elif audit_type == "rosbags_mcaps":
            issues = audit_rosbags_with_mcaps(name, processor_data, base_dir)
        elif audit_type == "models_rosbags_mcaps":
            issues = audit_models_with_rosbags_mcaps(name, processor_data, base_dir)
        elif audit_type == "models_rosbags_topics":
            issues = audit_models_with_rosbags_topics(name, processor_data, base_dir)
        else:
            continue

        all_issues.extend(issues)

    fixed = 0
    if do_fix and all_issues:
        fixed = fix_issues(completion_path, full_data, all_issues)

    return all_issues, fixed


def format_location(issue: Dict) -> str:
    """Format issue location as a readable string."""
    parts = []
    if "model" in issue:
        parts.append(issue["model"])
    parts.append(issue["rosbag"])
    if "mcap" in issue:
        parts.append(issue["mcap"])
    if "topic" in issue:
        parts.append(issue["topic"])
    return " / ".join(parts)


def main():
    do_fix = "--fix" in sys.argv

    config = Config.load_config()

    # All completion.json files and their processors
    audit_targets = [
        {
            "completion_path": config.topics_dir / "completion.json",
            "processors": [{
                "name": "topics_extraction_processor",
                "type": "rosbags_flat",
                "base_dir": config.topics_dir,
            }],
        },
        {
            "completion_path": config.lookup_tables_dir / "completion.json",
            "processors": [{
                "name": "timestamp_alignment_processor",
                "type": "rosbags_mcaps",
                "base_dir": config.lookup_tables_dir,
            }],
        },
        {
            "completion_path": config.positional_lookup_table_path.parent / "completion.json",
            "processors": [
                {
                    "name": "positional_lookup_processor",
                    "type": "rosbags_mcaps",
                    "base_dir": config.positional_lookup_table_path.parent,
                },
                {
                    "name": "positional_boundaries_processor",
                    "type": "rosbags_flat",
                    "base_dir": config.positional_lookup_table_path.parent,
                },
            ],
        },
        {
            "completion_path": config.image_topic_previews_dir / "completion.json",
            "processors": [{
                "name": "image_topic_previews_processor",
                "type": "rosbags_flat",
                "base_dir": config.image_topic_previews_dir,
            }],
        },
        {
            "completion_path": config.embeddings_dir / "completion.json",
            "processors": [{
                "name": "embeddings_processor",
                "type": "models_rosbags_mcaps",
                "base_dir": config.embeddings_dir,
            }],
        },
        {
            "completion_path": config.adjacent_similarities_dir / "completion.json",
            "processors": [{
                "name": "adjacent_similarities_postprocessor",
                "type": "models_rosbags_topics",
                "base_dir": config.adjacent_similarities_dir,
            }],
        },
    ]

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  COMPLETION.JSON AUDIT{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    if do_fix:
        print(f"  {YELLOW}Mode: FIX (broken entries will be removed){RESET}\n")
    else:
        print(f"  Mode: REPORT ONLY (use --fix to repair)\n")

    total_issues = 0
    total_fixed = 0

    for target in audit_targets:
        completion_path = target["completion_path"]
        processors = target["processors"]
        processor_names = ", ".join(p["name"] for p in processors)

        if not completion_path.exists():
            print(f"{CYAN}{processor_names}{RESET}")
            print(f"  {YELLOW}completion.json not found: {completion_path}{RESET}\n")
            continue

        issues, fixed = audit_completion_file(completion_path, processors, do_fix)

        if not issues:
            print(f"{CYAN}{processor_names}{RESET}")
            print(f"  {GREEN}OK{RESET}\n")
            continue

        # Group issues by processor
        by_processor = {}
        for issue in issues:
            by_processor.setdefault(issue["processor"], []).append(issue)

        for proc_name, proc_issues in by_processor.items():
            print(f"{CYAN}{proc_name}{RESET}  {RED}{len(proc_issues)} issue(s){RESET}")

            # Group by rosbag for compact display
            by_rosbag = {}
            for issue in proc_issues:
                key = issue["rosbag"]
                by_rosbag.setdefault(key, []).append(issue)

            for rosbag_name, rosbag_issues in sorted(by_rosbag.items()):
                if len(rosbag_issues) == 1:
                    issue = rosbag_issues[0]
                    print(f"  {rosbag_name}: {issue['detail']}")
                else:
                    print(f"  {rosbag_name}: {len(rosbag_issues)} entries with missing files")

        if do_fix and fixed > 0:
            print(f"  {GREEN}-> Fixed {fixed} entries{RESET}")

        total_issues += len(issues)
        total_fixed += fixed
        print()

    # Summary
    print(f"{BOLD}{'=' * 70}{RESET}")
    if total_issues == 0:
        print(f"  {GREEN}All completion.json files are consistent with disk.{RESET}")
    else:
        print(f"  {RED}Found {total_issues} issue(s) across all completion files.{RESET}")
        if do_fix:
            print(f"  {GREEN}Fixed {total_fixed} entries. Run pipeline to reprocess.{RESET}")
        else:
            print(f"  {YELLOW}Run with --fix to repair.{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")


if __name__ == "__main__":
    main()
