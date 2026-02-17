#!/usr/bin/env python3
"""CLI test runner for the MARSA research pipeline.

This script takes a query as input, runs the full research pipeline,
and prints the report with progress indicators.

Usage:
    python run_cli.py "What is the CAP theorem?"
    python run_cli.py --interactive
    python run_cli.py --hitl "Compare Rust vs Go"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import structlog  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.markdown import Markdown  # noqa: E402

from graph.workflow import run_research  # noqa: E402


console = Console()
logger = structlog.get_logger(__name__)


def print_header():
    """Print the MARSA header."""
    console.print()
    console.print("[bold blue]MARSA[/bold blue] - Multi-Agent Research Assistant", justify="center")
    console.print("[dim]Orchestrating AI agents for sourced research reports[/dim]", justify="center")
    console.print()


def print_progress(stage: str, detail: str = ""):
    """Print a progress indicator for the current stage.
    
    Args:
        stage: Current pipeline stage.
        detail: Additional detail to show.
    """
    stage_indicators = {
        "planning": "[cyan][PLANNING][/cyan]",
        "researching": "[green][RESEARCHING][/green]",
        "fact_checking": "[yellow][FACT-CHECKING][/yellow]",
        "synthesizing": "[magenta][SYNTHESIZING][/magenta]",
        "completed": "[bold green][COMPLETED][/bold green]",
        "failed": "[bold red][FAILED][/bold red]",
    }
    
    indicator = stage_indicators.get(stage.lower(), f"[{stage}]")
    
    if detail:
        console.print(f"  {indicator} {detail}")
    else:
        console.print(f"  {indicator}")


def print_trace_summary(state: dict):
    """Print a summary of the agent trace.
    
    Args:
        state: Final agent state with trace events.
    """
    trace = state.get("agent_trace", [])
    
    if not trace:
        return
    
    console.print()
    console.print("[bold]Agent Activity Summary[/bold]")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Agent", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Detail")
    table.add_column("Latency", justify="right")
    
    for event in trace[-15:]:  # Show last 15 events
        agent = getattr(event, 'agent', event.get('agent', 'unknown'))
        if hasattr(agent, 'value'):
            agent = agent.value
        
        action = getattr(event, 'action', event.get('action', ''))
        detail = getattr(event, 'detail', event.get('detail', ''))
        latency = getattr(event, 'latency_ms', event.get('latency_ms'))
        
        latency_str = f"{latency:.0f}ms" if latency else "-"
        
        # Truncate detail for display
        if len(detail) > 50:
            detail = detail[:47] + "..."
        
        table.add_row(str(agent), action, detail, latency_str)
    
    console.print(table)
    
    if len(trace) > 15:
        console.print(f"  [dim]... and {len(trace) - 15} more events[/dim]")


def print_report(state: dict):
    """Print the final research report.
    
    Args:
        state: Final agent state with report.
    """
    report_text = state.get("report", "")
    
    if not report_text:
        console.print("[red]No report generated.[/red]")
        return
    
    console.print()
    console.print(Panel(
        Markdown(report_text),
        title="[bold]Research Report[/bold]",
        border_style="blue",
        padding=(1, 2),
    ))


def print_citations(state: dict):
    """Print citation details.
    
    Args:
        state: Final agent state with citations.
    """
    citations = state.get("citations", [])
    
    if not citations:
        return
    
    console.print()
    console.print("[bold]Source Quality Details[/bold]")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Title")
    table.add_column("Quality", justify="center")
    table.add_column("URL", style="dim")
    
    for citation in citations:
        number = getattr(citation, 'number', citation.get('number', 0))
        title = getattr(citation, 'title', citation.get('title', ''))
        score = getattr(citation, 'source_quality_score', citation.get('source_quality_score', 0.5))
        url = getattr(citation, 'url', citation.get('url', ''))
        
        # Truncate for display
        if len(title) > 40:
            title = title[:37] + "..."
        if len(url) > 50:
            url = url[:47] + "..."
        
        # Color code quality
        if score >= 0.7:
            quality_str = f"[green]{score:.2f}[/green]"
        elif score >= 0.5:
            quality_str = f"[yellow]{score:.2f}[/yellow]"
        else:
            quality_str = f"[red]{score:.2f}[/red]"
        
        table.add_row(str(number), title, quality_str, url)
    
    console.print(table)


def print_stats(state: dict):
    """Print research statistics.
    
    Args:
        state: Final agent state.
    """
    report_structured = state.get("report_structured")
    
    if report_structured and hasattr(report_structured, 'metadata'):
        metadata = report_structured.metadata
        
        console.print()
        console.print("[bold]Statistics[/bold]")
        console.print(f"  Claims verified: {metadata.claims_verified}")
        console.print(f"  Fact-check pass rate: {metadata.fact_check_pass_rate:.1%}")
        console.print(f"  Sources consulted: {metadata.sources_searched}")
        console.print(f"  LLM calls: {metadata.llm_calls}")


async def run_with_progress(query: str, enable_hitl: bool = False) -> dict:
    """Run the research pipeline with live progress display.
    
    Args:
        query: The research query.
        enable_hitl: Whether to enable human-in-the-loop.
        
    Returns:
        Final agent state.
    """
    console.print(f"[bold]Query:[/bold] {query}")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=None)
        
        progress.update(task, description="[cyan]Planning research strategy...")
        
        # Run the research
        try:
            result = await run_research(query, enable_hitl=enable_hitl)
            progress.update(task, description="[green]Complete!")
            return result
        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}")
            raise


async def interactive_mode():
    """Run in interactive mode, accepting queries from stdin."""
    print_header()
    console.print("Enter your research queries below. Type 'quit' or 'exit' to stop.")
    console.print()
    
    while True:
        try:
            query = console.input("[bold cyan]Query>[/bold cyan] ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            
            console.print()
            
            try:
                result = await run_with_progress(query)
                print_trace_summary(result)
                print_report(result)
                print_citations(result)
                print_stats(result)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("research_error", error=str(e))
            
            console.print()
            console.print("-" * 60)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except EOFError:
            break


async def single_query(query: str, enable_hitl: bool = False, show_trace: bool = True):
    """Run a single query and display results.
    
    Args:
        query: The research query.
        enable_hitl: Whether to enable human-in-the-loop.
        show_trace: Whether to show agent trace summary.
    """
    print_header()
    
    try:
        result = await run_with_progress(query, enable_hitl=enable_hitl)
        
        if show_trace:
            print_trace_summary(result)
        
        print_report(result)
        print_citations(result)
        print_stats(result)
        
        # Check for errors
        errors = result.get("errors", [])
        if errors:
            console.print()
            console.print("[bold red]Errors encountered:[/bold red]")
            for error in errors:
                console.print(f"  - {error}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("research_error", error=str(e))
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="MARSA CLI - Multi-Agent Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is the CAP theorem?"
  %(prog)s --interactive
  %(prog)s --hitl "Compare Rust vs Go for distributed systems"
  %(prog)s --no-trace "Simple factual query"
        """,
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="The research query to investigate",
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    parser.add_argument(
        "--hitl",
        action="store_true",
        help="Enable human-in-the-loop checkpoints",
    )
    
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Don't show agent trace summary",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),  # DEBUG
        )
    else:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        )
    
    # Run appropriate mode
    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.query:
        asyncio.run(single_query(
            args.query,
            enable_hitl=args.hitl,
            show_trace=not args.no_trace,
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
