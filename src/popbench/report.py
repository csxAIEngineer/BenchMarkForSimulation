"""Write a JSON report and a short Markdown summary for one run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Summarize sample size, persona condition, and distance to the human ceiling."""
    lines = [
        "# Popbench report",
        "",
        f"- Sample size: {summary.get('n_people', 'unknown')}",
        f"- Persona condition: {summary.get('persona_condition', 'unknown')}",
        f"- Suite: {summary.get('suite', 'unknown')}",
        "",
        "These figures compare simulated answer shares with the Twin-2K human",
        "sample under the persona condition above. They are not evidence that the",
        "model can stand in for a human study.",
        "",
    ]
    ceiling = summary.get("test_retest_ceiling")
    if ceiling is None:
        lines.append("- Wave-4 test-retest ceiling: not computed for this run")
    else:
        lines.append(f"- Wave-4 test-retest ceiling: {ceiling}")
    distance = summary.get("distance_to_ceiling")
    if distance is not None:
        lines.append(f"- Distance to that ceiling: {distance}")
    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        lines.extend(["", "## Metrics", ""])
        for name, value in metrics.items():
            lines.append(f"- {name}: {value}")
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path, summary: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` under `run_dir`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(summary))
    return json_path, markdown_path
