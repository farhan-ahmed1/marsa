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

from graph.workflow import get_workflow, run_research  # noqa: E402
from graph.state import HITLFeedback, PipelineStatus, VerificationVerdict  # noqa: E402


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
        agent = getattr(event, 'agent', 'unknown')
        if hasattr(agent, 'value'):
            agent = agent.value
        
        action = getattr(event, 'action', '')
        detail = getattr(event, 'detail', '')
        latency = getattr(event, 'latency_ms', None)
        
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
        number = getattr(citation, 'number', 0)
        title = getattr(citation, 'title', '')
        score = getattr(citation, 'source_quality_score', 0.5)
        url = getattr(citation, 'url', '')
        
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


def _is_workflow_interrupted(state: dict) -> bool:
    """Check if the workflow is interrupted (waiting for HITL review).
    
    When HITL is enabled, the workflow interrupts after fact_checker.
    At that point, status is FACT_CHECKING and no report exists yet.
    """
    status = state.get("status", "")
    has_report = bool(state.get("report"))
    has_verification = bool(state.get("verification_results"))
    
    # Interrupted if we have verification results but no report and status is fact_checking
    return (
        status == PipelineStatus.FACT_CHECKING.value
        and has_verification
        and not has_report
    )


def _display_hitl_review(state: dict):
    """Display fact-check results for human review."""
    console.print()
    console.print(Panel(
        "[bold yellow]Human Review Required[/bold yellow]\n\n"
        "The fact-checker has verified the claims. Please review and decide how to proceed.",
        border_style="yellow",
    ))
    
    # Show verification summary
    verification_results = state.get("verification_results", [])
    if verification_results:
        console.print()
        console.print("[bold]Verification Results:[/bold]")
        
        supported = sum(
            1 for v in verification_results 
            if getattr(v, 'verdict', None) == VerificationVerdict.SUPPORTED
        )
        contradicted = sum(
            1 for v in verification_results 
            if getattr(v, 'verdict', None) == VerificationVerdict.CONTRADICTED
        )
        uncertain = len(verification_results) - supported - contradicted
        
        console.print(f"  Supported: [green]{supported}[/green]")
        console.print(f"  Contradicted: [red]{contradicted}[/red]")
        console.print(f"  Uncertain: [yellow]{uncertain}[/yellow]")


def _get_hitl_feedback() -> HITLFeedback:
    """Prompt the user for HITL feedback."""
    console.print()
    console.print("[bold]Options:[/bold]")
    console.print("  [cyan]1[/cyan] - Approve and continue to synthesis")
    console.print("  [cyan]2[/cyan] - Dig deeper into a specific topic")
    console.print("  [cyan]3[/cyan] - Abort research")
    console.print()
    
    while True:
        choice = console.input("[bold]Your choice (1-3):[/bold] ").strip()
        
        if choice == "1":
            return HITLFeedback(action="approve")
        elif choice == "2":
            topic = console.input("[bold]Topic to explore further:[/bold] ").strip()
            return HITLFeedback(action="dig_deeper", topic=topic)
        elif choice == "3":
            return HITLFeedback(action="abort")
        else:
            console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")


async def run_with_progress(query: str, enable_hitl: bool = False) -> dict:
    """Run the research pipeline with live progress display.
    
    Args:
        query: The research query.
        enable_hitl: Whether to enable human-in-the-loop.
        
    Returns:
        Final agent state.
    """
    import uuid
    
    console.print(f"[bold]Query:[/bold] {query}")
    console.print()
    
    thread_id = str(uuid.uuid4())
    result = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=None)
        
        progress.update(task, description="[cyan]Planning research strategy...")
        
        # Run the research
        result = await run_research(query, thread_id=thread_id, enable_hitl=enable_hitl)
        
        # Check if workflow is interrupted for HITL
        if enable_hitl and _is_workflow_interrupted(result):
            progress.update(task, description="[yellow]Awaiting human review...")
        else:
            progress.update(task, description="[green]Complete!")
    
    # Handle HITL review outside of progress context
    if enable_hitl and _is_workflow_interrupted(result):
        _display_hitl_review(result)
        feedback = _get_hitl_feedback()
        
        if feedback.action == "abort":
            console.print("[yellow]Research aborted by user.[/yellow]")
            return result
        
        # Resume workflow with feedback
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Resuming workflow...", total=None)
            
            # Get workflow and resume from checkpoint
            workflow = get_workflow(enable_hitl=enable_hitl)
            config = {"configurable": {"thread_id": thread_id}}
            
            # Inject feedback into the checkpointed state and resume
            # update_state adds values to the existing state, then ainvoke(None) resumes
            workflow.update_state(config, {"hitl_feedback": feedback})
            result = await workflow.ainvoke(None, config)
            progress.update(task, description="[green]Complete!")
    
    return result


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
