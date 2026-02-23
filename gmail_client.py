"""
Gmail API client with OAuth2 authentication.
Handles reading emails and performing actions (trash, label, etc.)
"""

import base64
import os
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from email.utils import parsedate_to_datetime

import structlog
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import Settings

logger = structlog.get_logger()

# Gmail API scopes - we need modify to trash emails
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify"
]


@dataclass
class Email:
    """Represents an email message."""
    id: str
    thread_id: str
    subject: str
    sender: str
    sender_email: str
    recipient: str
    date: datetime
    snippet: str
    body_preview: str
    labels: list[str]
    is_unread: bool
    
    def __str__(self):
        return f"[{self.date.strftime('%Y-%m-%d %H:%M')}] From: {self.sender} <{self.sender_email}> - {self.subject}"


class GmailClient:
    """Client for interacting with Gmail API."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Gmail using OAuth2."""
        creds = None
        
        if os.path.exists(self.settings.gmail_token_file):
            creds = Credentials.from_authorized_user_file(
                self.settings.gmail_token_file, SCOPES
            )
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired credentials")
                    creds.refresh(Request())
                except RefreshError as e:
                    logger.warning("Token refresh failed, re-authenticating", error=str(e))
                    creds = self._run_oauth_flow()
            else:
                creds = self._run_oauth_flow()
            
            self._save_token(creds)
        
        self._creds = creds
        self.service = build("gmail", "v1", credentials=creds)
        logger.info("Gmail API authenticated successfully")
    
    @staticmethod
    def _is_headless() -> bool:
        """Detect environments where no browser is available."""
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            return True
        if os.name == "posix" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
        try:
            webbrowser.get()
        except webbrowser.Error:
            return True
        return False
    
    def _run_oauth_flow(self) -> Credentials:
        """Run the full OAuth2 consent flow. Requires a browser."""
        if not os.path.exists(self.settings.gmail_credentials_file):
            raise FileNotFoundError(
                f"OAuth credentials file not found: {self.settings.gmail_credentials_file}\n"
                "Please download credentials.json from Google Cloud Console."
            )
        
        if self._is_headless():
            raise RuntimeError(
                "Gmail refresh token is expired or revoked and no browser is available "
                "to re-authenticate.\n"
                "Fix: use the 'Re-authenticate Gmail' button on the Vercel dashboard, "
                "or run 'python agent.py auth' locally and update the GOOGLE_TOKEN_JSON "
                "GitHub secret with the new token.json contents."
            )
        
        logger.info("Starting OAuth flow - browser will open for authentication")
        flow = InstalledAppFlow.from_client_secrets_file(
            self.settings.gmail_credentials_file, SCOPES
        )
        try:
            return flow.run_local_server(port=0)
        except webbrowser.Error as e:
            raise RuntimeError(
                f"Could not open browser for OAuth: {e}\n"
                "Fix: run 'python agent.py auth' on a machine with a browser, "
                "then copy the updated token.json here."
            ) from e
    
    def _save_token(self, creds: Credentials):
        """Persist credentials to disk so subsequent runs can reuse them."""
        with open(self.settings.gmail_token_file, "w") as token:
            token.write(creds.to_json())
        logger.info("Credentials saved", token_file=self.settings.gmail_token_file)
    
    def _refresh_and_retry(self, func, *args, **kwargs):
        """Re-authenticate and retry an API call once on auth failure."""
        try:
            self._creds.refresh(Request())
        except RefreshError:
            logger.warning("Refresh failed during retry, running full re-auth")
            self._creds = self._run_oauth_flow()
        
        self._save_token(self._creds)
        self.service = build("gmail", "v1", credentials=self._creds)
        return func(*args, **kwargs)
    
    def _persist_token_if_refreshed(self):
        """Save token to disk if the library auto-refreshed it in memory."""
        if self._creds and self._creds.valid and self._creds.token:
            self._save_token(self._creds)
    
    def _parse_email_address(self, header_value: str) -> tuple[str, str]:
        """Parse email header to extract name and email address."""
        if "<" in header_value and ">" in header_value:
            name = header_value.split("<")[0].strip().strip('"')
            email = header_value.split("<")[1].split(">")[0].strip()
        else:
            name = header_value.strip()
            email = header_value.strip()
        return name, email
    
    def _get_email_body_preview(self, payload: dict, max_length: int = 500) -> str:
        """Extract email body preview from message payload."""
        body = ""
        
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                        break
                elif "parts" in part:
                    # Handle nested parts
                    body = self._get_email_body_preview(part, max_length)
                    if body:
                        break
        elif "body" in payload:
            data = payload.get("body", {}).get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        
        # Clean and truncate
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        body = "\n".join(line.strip() for line in body.split("\n") if line.strip())
        
        if len(body) > max_length:
            body = body[:max_length] + "..."
        
        return body
    
    def get_unread_emails(self, max_results: int = 10, older_than: str | None = None) -> list[Email]:
        """
        Fetch unread emails from inbox.
        
        Args:
            max_results: Maximum number of emails to fetch
            older_than: Optional age filter (e.g., "1y", "6m", "30d", "2w")
                       Supported units: y (years), m (months), d (days), w (weeks)
        """
        try:
            return self._get_unread_emails_inner(max_results, older_than)
        except (HttpError, RefreshError) as e:
            if isinstance(e, HttpError) and e.resp.status != 401:
                logger.error("Failed to fetch emails", error=str(e))
                return []
            logger.warning("Auth expired during fetch, refreshing", error=str(e))
            return self._refresh_and_retry(self._get_unread_emails_inner, max_results, older_than)
    
    def _get_unread_emails_inner(self, max_results: int, older_than: str | None) -> list[Email]:
        query = "is:unread in:inbox"
        if older_than:
            query += f" older_than:{older_than}"
            logger.info("Filtering emails older than", older_than=older_than)
        
        results = self.service.users().messages().list(
            userId="me",
            q=query,
            maxResults=max_results
        ).execute()
        
        self._persist_token_if_refreshed()
        
        messages = results.get("messages", [])
        
        if not messages:
            logger.info("No unread emails found")
            return []
        
        emails = []
        for msg_ref in messages:
            try:
                email = self._fetch_email_details(msg_ref["id"])
                if email:
                    emails.append(email)
            except HttpError as e:
                logger.warning("Failed to fetch email details", error=str(e), message_id=msg_ref["id"])
        
        logger.info("Fetched unread emails", count=len(emails))
        return emails
    
    def _fetch_email_details(self, message_id: str) -> Optional[Email]:
        """Fetch full details of a single email."""
        msg = self.service.users().messages().get(
            userId="me",
            id=message_id,
            format="full"
        ).execute()
        
        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        
        # Parse sender
        sender_header = headers.get("from", "Unknown")
        sender_name, sender_email = self._parse_email_address(sender_header)
        
        # Parse date
        date_str = headers.get("date", "")
        try:
            email_date = parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            email_date = datetime.now()
        
        # Get body preview
        body_preview = self._get_email_body_preview(msg.get("payload", {}))
        
        # Get labels
        labels = msg.get("labelIds", [])
        
        return Email(
            id=message_id,
            thread_id=msg.get("threadId", ""),
            subject=headers.get("subject", "(No Subject)"),
            sender=sender_name,
            sender_email=sender_email.lower(),
            recipient=headers.get("to", ""),
            date=email_date,
            snippet=msg.get("snippet", ""),
            body_preview=body_preview,
            labels=labels,
            is_unread="UNREAD" in labels
        )
    
    def trash_email(self, email_id: str) -> bool:
        """Move an email to trash."""
        try:
            return self._trash_email_inner(email_id)
        except (HttpError, RefreshError) as e:
            if isinstance(e, HttpError) and e.resp.status != 401:
                logger.error("Failed to trash email", error=str(e), email_id=email_id)
                return False
            logger.warning("Auth expired during trash, refreshing", error=str(e))
            return self._refresh_and_retry(self._trash_email_inner, email_id)
    
    def _trash_email_inner(self, email_id: str) -> bool:
        self.service.users().messages().trash(
            userId="me",
            id=email_id
        ).execute()
        self._persist_token_if_refreshed()
        logger.info("Email moved to trash", email_id=email_id)
        return True
    
    def untrash_email(self, email_id: str) -> bool:
        """Restore an email from trash."""
        try:
            return self._untrash_email_inner(email_id)
        except (HttpError, RefreshError) as e:
            if isinstance(e, HttpError) and e.resp.status != 401:
                logger.error("Failed to untrash email", error=str(e), email_id=email_id)
                return False
            logger.warning("Auth expired during untrash, refreshing", error=str(e))
            return self._refresh_and_retry(self._untrash_email_inner, email_id)
    
    def _untrash_email_inner(self, email_id: str) -> bool:
        self.service.users().messages().untrash(
            userId="me",
            id=email_id
        ).execute()
        self._persist_token_if_refreshed()
        logger.info("Email restored from trash", email_id=email_id)
        return True
    
    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read."""
        try:
            return self._mark_as_read_inner(email_id)
        except (HttpError, RefreshError) as e:
            if isinstance(e, HttpError) and e.resp.status != 401:
                logger.error("Failed to mark as read", error=str(e), email_id=email_id)
                return False
            logger.warning("Auth expired during mark_as_read, refreshing", error=str(e))
            return self._refresh_and_retry(self._mark_as_read_inner, email_id)
    
    def _mark_as_read_inner(self, email_id: str) -> bool:
        self.service.users().messages().modify(
            userId="me",
            id=email_id,
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
        self._persist_token_if_refreshed()
        logger.info("Email marked as read", email_id=email_id)
        return True
    
    def add_label(self, email_id: str, label_name: str) -> bool:
        """Add a label to an email (creates label if needed)."""
        try:
            return self._add_label_inner(email_id, label_name)
        except (HttpError, RefreshError) as e:
            if isinstance(e, HttpError) and e.resp.status != 401:
                logger.error("Failed to add label", error=str(e), email_id=email_id)
                return False
            logger.warning("Auth expired during add_label, refreshing", error=str(e))
            return self._refresh_and_retry(self._add_label_inner, email_id, label_name)
    
    def _add_label_inner(self, email_id: str, label_name: str) -> bool:
        labels_result = self.service.users().labels().list(userId="me").execute()
        label_id = None
        
        for label in labels_result.get("labels", []):
            if label["name"].lower() == label_name.lower():
                label_id = label["id"]
                break
        
        if not label_id:
            new_label = self.service.users().labels().create(
                userId="me",
                body={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
            ).execute()
            label_id = new_label["id"]
            logger.info("Created new label", label_name=label_name)
        
        self.service.users().messages().modify(
            userId="me",
            id=email_id,
            body={"addLabelIds": [label_id]}
        ).execute()
        
        self._persist_token_if_refreshed()
        logger.info("Label added to email", email_id=email_id, label=label_name)
        return True
    
    def get_user_email(self) -> str:
        """Get the authenticated user's email address."""
        try:
            return self._get_user_email_inner()
        except (HttpError, RefreshError) as e:
            if isinstance(e, HttpError) and e.resp.status != 401:
                logger.error("Failed to get user profile", error=str(e))
                return ""
            logger.warning("Auth expired during get_user_email, refreshing", error=str(e))
            return self._refresh_and_retry(self._get_user_email_inner)
    
    def _get_user_email_inner(self) -> str:
        profile = self.service.users().getProfile(userId="me").execute()
        self._persist_token_if_refreshed()
        return profile.get("emailAddress", "")

