"""Production Structured Telemetry & Metrics Logger.

Records every single LLM API call with:
- Timestamp & latency (ms)
- Token consumption (prompt, completion, total)
- Model, stage, and fallback status
- Success/failure status & error strings
- Immediate persistence to JSONL (crash-safe)
- Generation of run summary JSON and Markdown reports
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CallRecord:
    """Detailed record for an individual LLM API invocation."""
    call_index: int
    timestamp: str
    stage: str
    model: str
    is_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    latency_sec: float
    success: bool
    error: Optional[str] = None
    input_preview: str = ""
    output_preview: str = ""


class PipelineLogger:
    """Manages telemetry, metrics aggregation, and audit logging per pipeline run."""

    def __init__(self, log_dir: Union[str, Path] = "output/logs", run_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"run_{self.run_id}.jsonl"
        self.summary_path = self.log_dir / f"run_{self.run_id}_summary.json"
        self.report_path = self.log_dir / f"run_{self.run_id}_report.md"

        self.records: List[CallRecord] = []
        self._start_time = time.time()
        self._start_iso = datetime.now().isoformat()

        # Aggregators
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.fallback_calls = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.latencies_ms: List[float] = []

        self.calls_by_stage: Dict[str, int] = {}
        self.tokens_by_stage: Dict[str, Dict[str, int]] = {}
        self.latency_by_stage: Dict[str, List[float]] = {}
        self.calls_by_model: Dict[str, int] = {}

        print(f"📝 [Telemetry] Logging calls to: {self.log_path}")

    def log_call(
        self,
        stage: str,
        model: str,
        is_fallback: bool,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        input_preview: str = "",
        output_preview: str = "",
    ) -> CallRecord:
        """Record and immediately flush an API call."""
        self.total_calls += 1
        record = CallRecord(
            call_index=self.total_calls,
            timestamp=datetime.now().isoformat(),
            stage=stage,
            model=model,
            is_fallback=is_fallback,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            latency_sec=round(latency_ms / 1000.0, 3),
            success=success,
            error=error,
            input_preview=input_preview[:120].replace("\n", " "),
            output_preview=output_preview[:150].replace("\n", " "),
        )
        self.records.append(record)

        # Update stats
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        if is_fallback:
            self.fallback_calls += 1

        self.total_tokens += total_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.latencies_ms.append(latency_ms)

        # Stage breakdown
        self.calls_by_stage[stage] = self.calls_by_stage.get(stage, 0) + 1
        if stage not in self.tokens_by_stage:
            self.tokens_by_stage[stage] = {"prompt": 0, "completion": 0, "total": 0}
        self.tokens_by_stage[stage]["prompt"] += prompt_tokens
        self.tokens_by_stage[stage]["completion"] += completion_tokens
        self.tokens_by_stage[stage]["total"] += total_tokens

        if stage not in self.latency_by_stage:
            self.latency_by_stage[stage] = []
        self.latency_by_stage[stage].append(latency_ms)

        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

        # Crash-safe write immediately
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        return record

    def save_summary(self) -> Dict[str, Any]:
        """Aggregate all metrics and persist run summary JSON and Markdown report."""
        elapsed = time.time() - self._start_time
        avg_latency = sum(self.latencies_ms) / max(len(self.latencies_ms), 1)
        min_latency = min(self.latencies_ms) if self.latencies_ms else 0.0
        max_latency = max(self.latencies_ms) if self.latencies_ms else 0.0

        stage_metrics = {}
        for stage, count in self.calls_by_stage.items():
            lats = self.latency_by_stage.get(stage, [0.0])
            stage_metrics[stage] = {
                "calls": count,
                "total_tokens": self.tokens_by_stage[stage]["total"],
                "prompt_tokens": self.tokens_by_stage[stage]["prompt"],
                "completion_tokens": self.tokens_by_stage[stage]["completion"],
                "avg_latency_ms": round(sum(lats) / max(len(lats), 1), 1),
            }

        summary = {
            "run_id": self.run_id,
            "start_time": self._start_iso,
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
            "calls": {
                "total": self.total_calls,
                "successful": self.successful_calls,
                "failed": self.failed_calls,
                "fallback": self.fallback_calls,
            },
            "tokens": {
                "total": self.total_tokens,
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "avg_per_call": round(self.total_tokens / max(self.total_calls, 1), 1),
            },
            "latency_ms": {
                "avg": round(avg_latency, 1),
                "min": round(min_latency, 1),
                "max": round(max_latency, 1),
            },
            "calls_by_model": self.calls_by_model,
            "stage_breakdown": stage_metrics,
            "artifacts": {
                "jsonl_log": str(self.log_path),
                "summary_json": str(self.summary_path),
                "report_md": str(self.report_path),
            },
        }

        # Write summary JSON
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Write Markdown Report
        self._write_markdown_report(summary)

        return summary

    def _write_markdown_report(self, s: Dict[str, Any]) -> None:
        """Write a clean, readable Markdown telemetry report."""
        report = [
            f"# Run Telemetry Report — `{s['run_id']}`",
            "",
            f"- **Start Time:** {s['start_time']}",
            f"- **Duration:** {s['elapsed_human']} ({s['elapsed_seconds']}s)",
            f"- **Total Calls:** {s['calls']['total']} (Success: {s['calls']['successful']}, Fallback: {s['calls']['fallback']}, Failed: {s['calls']['failed']})",
            f"- **Total Tokens:** {s['tokens']['total']:,} (Prompt: {s['tokens']['prompt']:,}, Completion: {s['tokens']['completion']:,})",
            f"- **Latency (ms):** Avg: {s['latency_ms']['avg']}ms | Min: {s['latency_ms']['min']}ms | Max: {s['latency_ms']['max']}ms",
            "",
            "## Stage Breakdown",
            "",
            "| Stage | Calls | Total Tokens | Prompt Tokens | Comp. Tokens | Avg Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for stage, data in s["stage_breakdown"].items():
            report.append(
                f"| `{stage}` | {data['calls']} | {data['total_tokens']:,} | {data['prompt_tokens']:,} | {data['completion_tokens']:,} | {data['avg_latency_ms']:.1f} |"
            )

        report.extend([
            "",
            "## Model Invocations",
            "",
            "| Model | Call Count |",
            "| :--- | :---: |",
        ])
        for model, cnt in s["calls_by_model"].items():
            report.append(f"| `{model}` | {cnt} |")

        report.extend([
            "",
            f"*(Detailed per-call log available at `{self.log_path}`)*",
        ])

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")

    def print_summary(self) -> None:
        """Print formatted terminal report."""
        s = self.save_summary()
        print("\n" + "═" * 70)
        print(f"📊 PIPELINE RUN TELEMETRY & LOGGING (Run ID: {s['run_id']})")
        print("═" * 70)
        print(f"  ⏱️  Duration:        {s['elapsed_human']} ({s['elapsed_seconds']}s)")
        print(f"  📞 Total Calls:     {s['calls']['total']} (✅ {s['calls']['successful']} ok, 🔄 {s['calls']['fallback']} fallback, ❌ {s['calls']['failed']} fail)")
        print(f"  🎟️  Total Tokens:    {s['tokens']['total']:,} (Prompt: {s['tokens']['prompt']:,}, Completion: {s['tokens']['completion']:,})")
        print(f"  ⚡ Latency:         Avg {s['latency_ms']['avg']} ms  |  Min {s['latency_ms']['min']} ms  |  Max {s['latency_ms']['max']} ms")
        print(f"  🎯 Avg Tokens/Call: {s['tokens']['avg_per_call']:.1f}")
        print("\n  STAGE BREAKDOWN:")
        print(f"  {'Stage':<24s} {'Calls':>6s} {'Tokens':>10s} {'Avg Latency':>14s}")
        print("  " + "─" * 58)
        for stage, d in s["stage_breakdown"].items():
            print(f"  {stage:<24s} {d['calls']:>6d} {d['total_tokens']:>10,d} {d['avg_latency_ms']:>12.1f} ms")
        print("\n  📁 LOG ARTIFACTS:")
        print(f"  - Per-call Log:  {self.log_path}")
        print(f"  - Run Summary:   {self.summary_path}")
        print(f"  - Run Report:    {self.report_path}")
        print("═" * 70 + "\n")
