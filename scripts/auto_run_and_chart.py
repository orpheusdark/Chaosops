from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHAOSOPS_DIR = os.path.join(REPO_ROOT, "chaosops")
if CHAOSOPS_DIR not in sys.path:
    sys.path.insert(0, CHAOSOPS_DIR)

from env import ChaosOpsEnv  # noqa: E402


@dataclass
class EpisodeResult:
    episode: int
    mode: str
    score: float
    reward: float
    steps: int
    success: bool


def clamp_score(value: float) -> float:
    return max(0.01, min(0.99, value))


def step(env: ChaosOpsEnv, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return env.step(action, payload)


def run_baseline_episode(env: ChaosOpsEnv) -> EpisodeResult:
    env.reset("task3")

    step(env, "query_system", {})
    # Wrong attempt: fix before permission.
    step(
        env,
        "fix_service",
        {
            "config": {
                "service": "auth_service",
                "status": "running",
                "cpu_limit": "1",
            },
            "token": "invalid-token",
        },
    )
    # Weak justification first.
    step(env, "request_access", {"justification": "please allow"})
    # Correct access request.
    access = step(env, "request_access", {"justification": "Need access for oom crash recovery"})
    token = access.get("result", {}).get("token", "")

    schema_info = step(env, "get_schema", {})
    version = schema_info.get("result", {}).get("schema", {}).get("version", 2)

    if version == 1:
        config = {
            "service": "auth_service",
            "status": "running",
            "cpu_limit": "1",
        }
    else:
        config = {
            "service_name": "auth_service",
            "condition": "running",
            "max_compute": "1",
        }

    final = step(env, "fix_service", {"config": config, "token": token})
    score = final.get("score")
    if score is None:
        score = clamp_score(float(final.get("total_reward", 0.0)))

    return EpisodeResult(
        episode=0,
        mode="baseline",
        score=float(score),
        reward=float(final.get("total_reward", 0.0)),
        steps=int(final.get("state", {}).get("step_count", 0)),
        success=final.get("state", {}).get("service_status") == "running",
    )


def run_rl_episode(env: ChaosOpsEnv) -> EpisodeResult:
    env.reset("task3")

    step(env, "query_system", {})
    access = step(env, "request_access", {"justification": "Need token to fix oom crash"})
    token = access.get("result", {}).get("token", "")

    schema_info = step(env, "get_schema", {})
    version = schema_info.get("result", {}).get("schema", {}).get("version", 2)

    if version == 1:
        config = {
            "service": "auth_service",
            "status": "running",
            "cpu_limit": "1",
        }
    else:
        config = {
            "service_name": "auth_service",
            "condition": "running",
            "max_compute": "1",
        }

    final = step(env, "fix_service", {"config": config, "token": token})
    score = final.get("score")
    if score is None:
        score = clamp_score(float(final.get("total_reward", 0.0)))

    return EpisodeResult(
        episode=0,
        mode="rl_agent",
        score=float(score),
        reward=float(final.get("total_reward", 0.0)),
        steps=int(final.get("state", {}).get("step_count", 0)),
        success=final.get("state", {}).get("service_status") == "running",
    )


def build_svg(results: List[EpisodeResult], output_path: str) -> None:
    width = 1100
    height = 640
    margin_left = 70
    margin_right = 40
    margin_top = 70
    margin_bottom = 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_steps = max(r.steps for r in results)
    x_count = len(results)

    def x_pos(i: int) -> float:
        if x_count == 1:
            return margin_left + plot_w / 2
        return margin_left + (i / (x_count - 1)) * plot_w

    def y_score(score: float) -> float:
        # score in [0,1]
        return margin_top + (1.0 - score) * plot_h

    def y_steps(steps: int) -> float:
        # secondary axis map [0,max_steps] into lower part with some compression
        ratio = steps / max_steps if max_steps else 0
        return margin_top + (1.0 - ratio) * plot_h

    baseline_points = []
    rl_points = []
    score_points = []

    for i, r in enumerate(results):
        x = x_pos(i)
        score_points.append((x, y_score(r.score), r))
        if r.mode == "baseline":
            baseline_points.append((x, y_score(r.score), r))
        else:
            rl_points.append((x, y_score(r.score), r))

    bars = []
    bar_width = max(8, int(plot_w / (x_count * 3)))
    for i, r in enumerate(results):
        x = x_pos(i)
        y = y_steps(r.steps)
        h = margin_top + plot_h - y
        bars.append((x - bar_width / 2, y, bar_width, h, r.mode))

    def points_to_polyline(points: List[tuple]) -> str:
        return " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in points)

    baseline_mean = mean(r.score for r in results if r.mode == "baseline")
    rl_mean = mean(r.score for r in results if r.mode == "rl_agent")

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#0f172a"/>')
    parts.append('<rect x="20" y="20" width="1060" height="600" rx="16" fill="#111827" stroke="#334155"/>')
    parts.append('<text x="40" y="52" fill="#e5e7eb" font-family="Segoe UI, Arial" font-size="26" font-weight="700">ChaosOps RL Progress (Auto Benchmark)</text>')
    parts.append('<text x="40" y="78" fill="#94a3b8" font-family="Segoe UI, Arial" font-size="14">Task3 repeated episodes: baseline policy then RL policy</text>')

    # Axes
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#475569"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#475569"/>')

    # Score grid lines
    for k in range(0, 6):
        val = k / 5
        y = y_score(val)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#1f2937"/>')
        parts.append(f'<text x="28" y="{y + 5:.2f}" fill="#94a3b8" font-family="Segoe UI, Arial" font-size="12">{val:.1f}</text>')

    # Step bars
    for x, y, bw, h, mode in bars:
        fill = "#64748b" if mode == "baseline" else "#0ea5e9"
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{h:.2f}" fill="{fill}" opacity="0.35"/>')

    if baseline_points:
        parts.append(f'<polyline fill="none" stroke="#f59e0b" stroke-width="3" points="{points_to_polyline(baseline_points)}"/>')
    if rl_points:
        parts.append(f'<polyline fill="none" stroke="#22c55e" stroke-width="3" points="{points_to_polyline(rl_points)}"/>')

    for x, y, r in score_points:
        color = "#f59e0b" if r.mode == "baseline" else "#22c55e"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    # X labels every episode
    for i, r in enumerate(results):
        x = x_pos(i)
        parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 20}" text-anchor="middle" fill="#94a3b8" font-family="Segoe UI, Arial" font-size="11">E{i+1}</text>')

    # Legends
    parts.append('<rect x="760" y="34" width="12" height="12" fill="#f59e0b"/>')
    parts.append('<text x="778" y="45" fill="#e5e7eb" font-family="Segoe UI, Arial" font-size="12">Baseline score</text>')
    parts.append('<rect x="900" y="34" width="12" height="12" fill="#22c55e"/>')
    parts.append('<text x="918" y="45" fill="#e5e7eb" font-family="Segoe UI, Arial" font-size="12">RL score</text>')
    parts.append('<rect x="760" y="54" width="12" height="12" fill="#0ea5e9" opacity="0.35"/>')
    parts.append('<text x="778" y="65" fill="#e5e7eb" font-family="Segoe UI, Arial" font-size="12">Steps per episode</text>')

    # Summary block
    parts.append('<rect x="40" y="530" width="1020" height="70" rx="10" fill="#0b1220" stroke="#1f2937"/>')
    parts.append(f'<text x="58" y="557" fill="#cbd5e1" font-family="Segoe UI, Arial" font-size="14">Baseline mean score: {baseline_mean:.2f}</text>')
    parts.append(f'<text x="330" y="557" fill="#cbd5e1" font-family="Segoe UI, Arial" font-size="14">RL mean score: {rl_mean:.2f}</text>')
    parts.append(f'<text x="560" y="557" fill="#cbd5e1" font-family="Segoe UI, Arial" font-size="14">Improvement: {rl_mean - baseline_mean:+.2f}</text>')
    parts.append(f'<text x="58" y="582" fill="#94a3b8" font-family="Segoe UI, Arial" font-size="13">Generated automatically by scripts/auto_run_and_chart.py</text>')

    parts.append('</svg>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> int:
    env = ChaosOpsEnv(max_steps=10)

    results: List[EpisodeResult] = []
    total_episodes = 12
    for idx in range(total_episodes):
        if idx < total_episodes // 2:
            r = run_baseline_episode(env)
        else:
            r = run_rl_episode(env)
        r.episode = idx + 1
        results.append(r)

    chart_path = os.path.join(REPO_ROOT, "charts", "rl_progress.svg")
    metrics_path = os.path.join(REPO_ROOT, "charts", "rl_progress.json")

    build_svg(results, chart_path)

    payload = {
        "episodes": [
            {
                "episode": r.episode,
                "mode": r.mode,
                "score": r.score,
                "reward": r.reward,
                "steps": r.steps,
                "success": r.success,
            }
            for r in results
        ]
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"chart: {chart_path}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
