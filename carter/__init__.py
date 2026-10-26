"""
Carter — Fully Offline Personal AI Agent
Entry point: python -m carter
"""

import sys
import click
from carter.runtime.readiness import run_readiness_check
from carter.utils.config import load_config
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
    """Run the readiness check and report component status."""
    cfg = ctx.obj["config"]
    run_readiness_check(cfg)


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Start the Carter interactive agent loop."""
    from carter.agent.loop import AgentLoop
    cfg = ctx.obj["config"]
    loop = AgentLoop(cfg)
    loop.start()


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
