"""
Email Agent - Autonomous email classifier and organizer.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import Settings, get_settings
from gmail_client import GmailClient, Email
from classifier import EmailClassifier, ClassificationResult, ImportanceLevel

logger = structlog.get_logger()
console = Console()


@dataclass
class ActionLog:
    """Log of an action taken by the agent."""
    timestamp: str
    email_id: str
    email_subject: str
    email_sender: str
    classification: str
    confidence: float
    action: str
    success: bool
    reasoning: str


class EmailAgent:
    """Autonomous email classification and organization agent."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gmail = GmailClient(settings)
        self.classifier = EmailClassifier(settings)
        self.action_log: list[ActionLog] = []
        self.log_file = "agent_actions.json"
        
        # Load previous action log
        self._load_action_log()
    
    def _load_action_log(self):
        """Load action log from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    self.action_log = [ActionLog(**item) for item in data]
            except (json.JSONDecodeError, TypeError):
                self.action_log = []
    
    def _save_action_log(self):
        """Save action log to file."""
        with open(self.log_file, "w") as f:
            json.dump([asdict(log) for log in self.action_log], f, indent=2)
    
    def _log_action(
        self,
        email: Email,
        classification: ClassificationResult,
        action: str,
        success: bool
    ):
        """Log an action taken by the agent."""
        log_entry = ActionLog(
            timestamp=datetime.now().isoformat(),
            email_id=email.id,
            email_subject=email.subject[:100],
            email_sender=email.sender_email,
            classification=classification.importance.value,
            confidence=classification.confidence,
            action=action,
            success=success,
            reasoning=classification.reasoning
        )
        self.action_log.append(log_entry)
        self._save_action_log()
    
    def _should_auto_trash(self, classification: ClassificationResult) -> bool:
        """Determine if email should be auto-trashed based on classification."""
        return (
            classification.importance == ImportanceLevel.NOT_IMPORTANT and
            classification.confidence >= self.settings.confidence_threshold and
            classification.suggested_action == "trash"
        )
    
    def process_emails(self, interactive: bool = True, older_than: str | None = None) -> dict:
        """
        Process unread emails - classify and take action.
        
        Args:
            interactive: If True, show results and ask for confirmation
            older_than: Optional age filter (e.g., "1y", "6m", "30d")
        
        Returns:
            Summary statistics
        """
        stats = {
            "processed": 0,
            "important": 0,
            "not_important": 0,
            "uncertain": 0,
            "trashed": 0,
            "kept": 0,
            "errors": 0
        }
        
        # Fetch unread emails
        console.print("\n[bold blue]📬 Fetching unread emails...[/bold blue]")
        if older_than:
            console.print(f"[dim]Filtering: older than {older_than}[/dim]")
        emails = self.gmail.get_unread_emails(max_results=self.settings.batch_size, older_than=older_than)
        
        if not emails:
            console.print("[yellow]No unread emails found.[/yellow]")
            return stats
        
        console.print(f"[green]Found {len(emails)} unread email(s)[/green]\n")
        
        # Classify all emails
        console.print("[bold blue]🤖 Classifying emails...[/bold blue]\n")
        classifications = []
        
        # Use batch classification (10 emails per API call)
        batch_results = self.classifier.classify_batch(emails, batch_size=10)
        
        # Pair emails with their classifications
        classifications = list(zip(emails, batch_results))
        
        # Update stats
        for email, classification in classifications:
            stats["processed"] += 1
            if classification.importance == ImportanceLevel.IMPORTANT:
                stats["important"] += 1
            elif classification.importance == ImportanceLevel.NOT_IMPORTANT:
                stats["not_important"] += 1
            else:
                stats["uncertain"] += 1
        
        # Display results
        self._display_classification_results(classifications)
        
        # Handle actions
        if self.settings.dry_run:
            console.print(Panel(
                "[yellow]DRY RUN MODE[/yellow] - No emails will be trashed.\n"
                "Set DRY_RUN=false in .env to enable automatic actions.",
                title="⚠️ Dry Run",
                box=box.ROUNDED
            ))
        
        # Process each classification
        for email, classification in classifications:
            if self._should_auto_trash(classification):
                if self.settings.dry_run:
                    console.print(f"  [dim]Would trash: {email.subject[:50]}...[/dim]")
                    self._log_action(email, classification, "would_trash", True)
                else:
                    success = self.gmail.trash_email(email.id)
                    if success:
                        stats["trashed"] += 1
                        console.print(f"  [red]🗑️ Trashed: {email.subject[:50]}...[/red]")
                    else:
                        stats["errors"] += 1
                    self._log_action(email, classification, "trash", success)
            else:
                stats["kept"] += 1
                self._log_action(email, classification, "keep", True)
        
        # Display summary
        self._display_summary(stats)
        
        return stats
    
    def _display_classification_results(self, classifications: list[tuple[Email, ClassificationResult]]):
        """Display classification results in a formatted table."""
        table = Table(title="📧 Email Classification Results", box=box.ROUNDED)
        table.add_column("From", style="cyan", max_width=25)
        table.add_column("Subject", style="white", max_width=35)
        table.add_column("Category", style="magenta")
        table.add_column("Importance", style="bold")
        table.add_column("Confidence", justify="right")
        table.add_column("Action", style="yellow")
        
        for email, classification in classifications:
            # Color-code importance
            if classification.importance == ImportanceLevel.IMPORTANT:
                importance_style = "[green]✓ Important[/green]"
            elif classification.importance == ImportanceLevel.NOT_IMPORTANT:
                importance_style = "[red]✗ Not Important[/red]"
            else:
                importance_style = "[yellow]? Uncertain[/yellow]"
            
            # Confidence bar
            conf_pct = int(classification.confidence * 100)
            conf_display = f"{conf_pct}%"
            
            # Suggested action
            action = classification.suggested_action.upper()
            if action == "TRASH":
                action = "[red]TRASH[/red]"
            elif action == "KEEP":
                action = "[green]KEEP[/green]"
            else:
                action = "[yellow]REVIEW[/yellow]"
            
            table.add_row(
                email.sender[:25],
                email.subject[:35],
                classification.category,
                importance_style,
                conf_display,
                action
            )
        
        console.print(table)
        console.print()
    
    def _display_summary(self, stats: dict):
        """Display processing summary."""
        summary = f"""
[bold]Processing Summary[/bold]
━━━━━━━━━━━━━━━━━━━━
📧 Processed: {stats['processed']}
✅ Important: {stats['important']}
❌ Not Important: {stats['not_important']}
❓ Uncertain: {stats['uncertain']}
🗑️  Trashed: {stats['trashed']}
📥 Kept: {stats['kept']}
⚠️  Errors: {stats['errors']}
"""
        console.print(Panel(summary, title="📊 Summary", box=box.ROUNDED))
    
    def undo_last_trash(self) -> bool:
        """Undo the last trash action."""
        # Find last trash action
        for log in reversed(self.action_log):
            if log.action == "trash" and log.success:
                success = self.gmail.untrash_email(log.email_id)
                if success:
                    console.print(f"[green]✓ Restored: {log.email_subject}[/green]")
                    return True
                else:
                    console.print("[red]Failed to restore email[/red]")
                    return False
        
        console.print("[yellow]No recent trash action found to undo[/yellow]")
        return False
    
    def show_action_history(self, limit: int = 20):
        """Display recent action history."""
        if not self.action_log:
            console.print("[yellow]No action history found[/yellow]")
            return
        
        table = Table(title="📜 Action History", box=box.ROUNDED)
        table.add_column("Time", style="dim")
        table.add_column("Sender", style="cyan")
        table.add_column("Subject", max_width=30)
        table.add_column("Classification")
        table.add_column("Action", style="yellow")
        
        for log in self.action_log[-limit:]:
            timestamp = log.timestamp.split("T")[1].split(".")[0]
            table.add_row(
                timestamp,
                log.email_sender[:20],
                log.email_subject[:30],
                log.classification,
                log.action
            )
        
        console.print(table)


def main():
    """Main entry point."""
    import typer
    
    app = typer.Typer(help="Email Agent - Autonomous email classifier and organizer")
    
    @app.command()
    def run(
        batch_size: int = typer.Option(10, "--batch", "-b", help="Number of emails to process"),
        dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Don't actually trash emails"),
        older_than: str = typer.Option(None, "--older-than", "-o", help="Only process emails older than (e.g., 1y, 6m, 30d, 2w)"),
    ):
        """Process unread emails and classify them."""
        settings = get_settings()
        settings.batch_size = batch_size
        settings.dry_run = dry_run
        
        console.print(Panel(
            "[bold blue]🤖 Email Agent[/bold blue]\n"
            "Autonomous email classification and organization",
            box=box.DOUBLE
        ))
        
        agent = EmailAgent(settings)
        
        # Show user info
        user_email = agent.gmail.get_user_email()
        console.print(f"[dim]Logged in as: {user_email}[/dim]\n")
        
        agent.process_emails(older_than=older_than)
    
    @app.command()
    def undo():
        """Undo the last trash action."""
        settings = get_settings()
        agent = EmailAgent(settings)
        agent.undo_last_trash()
    
    @app.command()
    def history(limit: int = typer.Option(20, "--limit", "-l", help="Number of entries to show")):
        """Show action history."""
        settings = get_settings()
        agent = EmailAgent(settings)
        agent.show_action_history(limit)
    
    @app.command()
    def auth():
        """Authenticate with Gmail (run this first)."""
        settings = get_settings()
        console.print("[bold blue]Starting Gmail authentication...[/bold blue]")
        GmailClient(settings)
        console.print("[green]✓ Authentication successful![/green]")
    
    app()


if __name__ == "__main__":
    main()

