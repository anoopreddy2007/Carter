"""
Carter — Fully Offline Personal AI Agent
CLI Entry Point
"""

from __future__ import annotations

import json
from pathlib import Path
import click

from carter.runtime.readiness import run_readiness_check
from carter.utils.config import load_config, get
from carter.utils.logging import setup_logging


@click.group()
@click.option("--config", default="configs/carter.yaml", help="Path to configuration file.")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Carter: your fully offline AI agent."""
    cfg = load_config(config)
    setup_logging(cfg)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@cli.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Run readiness checks and verify all offline components."""
    cfg = ctx.obj["config"]
    ready = run_readiness_check(cfg)
    if not ready:
        raise click.ClickException("Carter readiness check failed. Review messages above.")


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Start the interactive Carter agent REPL."""
    from carter.agent.loop import AgentLoop
    cfg = ctx.obj["config"]
    loop = AgentLoop(cfg)
    loop.start()


@cli.command()
@click.argument("request")
@click.pass_context
def ask(ctx: click.Context, request: str) -> None:
    """Run a single request through Carter and print the final answer."""
    from carter.agent.loop import AgentLoop
    cfg = ctx.obj["config"]
    loop = AgentLoop(cfg)
    res = loop.run_task(request)
    click.echo(res.answer)


@cli.command("train-ml")
@click.option("--samples", default=1500, help="Number of telemetry samples to generate.")
@click.option("--contamination", default=0.05, help="Anomaly ratio.")
@click.option("--output-dir", default="artifacts/models", help="Directory to save model artifacts.")
@click.pass_context
def train_ml(ctx: click.Context, samples: int, contamination: float, output_dir: str) -> None:
    """Train the offline telemetry anomaly detection model."""
    from carter.ml.training.train import train_anomaly_model
    click.echo(f"Training offline anomaly detection model ({samples} samples)...")
    meta = train_anomaly_model(
        output_dir=output_dir,
        num_samples=samples,
        contamination=contamination,
    )
    f1 = meta["metrics"]["f1"]
    click.echo(f"Model saved to: {meta['model_artifact']}")
    click.echo(f"Evaluation F1 Score: {f1:.4f}")


@cli.command("audit")
@click.option("--limit", default=20, help="Number of recent records to display.")
@click.pass_context
def audit(ctx: click.Context, limit: int) -> None:
    """View recent actions recorded in the local audit log."""
    from carter.agent.audit import create_audit_logger
    cfg = ctx.obj["config"]
    audit_logger = create_audit_logger(cfg)
    records = audit_logger.read_records(limit=limit)

    if not records:
        click.echo("No audit records found.")
        return

    click.echo(f"--- Recent Carter Audit Trail ({len(records)} entries) ---")
    for rec in records:
        ts = rec.get("timestamp", "")[:19]
        if rec.get("event") == "task_completed":
            status = "SUCCESS" if rec.get("success") else "FAILED"
            click.echo(f"[{ts}] TASK COMPLETED: {status} in {rec.get('steps_used')} steps ({rec.get('total_duration_ms'):.1f}ms)")
        else:
            tool = rec.get("tool_name", "")
            duration = rec.get("duration_ms", 0.0)
            err = f" | ERROR: {rec['error']}" if rec.get("error") else ""
            click.echo(f"[{ts}] STEP #{rec.get('step_num')}: {tool} ({duration:.1f}ms){err}")


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
