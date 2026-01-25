"""
Gmail API client with OAuth2 authentication.
Handles reading emails and performing actions (trash, label, etc.)
"""

import base64
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from email.utils import parsedate_to_datetime

import structlog
from google.auth.transport.requests import Request
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
        
        # Check for existing token
        if os.path.exists(self.settings.gmail_token_file):
            creds = Credentials.from_authorized_user_file(
                self.settings.gmail_token_file, SCOPES
            )
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
            else:
                if not os.path.exists(self.settings.gmail_credentials_file):
                    raise FileNotFoundError(
                        f"OAuth credentials file not found: {self.settings.gmail_credentials_file}\n"
                        "Please download credentials.json from Google Cloud Console."
                    )
                
                logger.info("Starting OAuth flow - browser will open for authentication")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.settings.gmail_credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.settings.gmail_token_file, "w") as token:
                token.write(creds.to_json())
            logger.info("Credentials saved", token_file=self.settings.gmail_token_file)
        
        self.service = build("gmail", "v1", credentials=creds)
        logger.info("Gmail API authenticated successfully")
    
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
    
    def get_unread_emails(self, max_results: int = 10) -> list[Email]:
        """Fetch unread emails from inbox."""
        try:
            # Search for unread emails in inbox
            results = self.service.users().messages().list(
                userId="me",
                q="is:unread in:inbox",
                maxResults=max_results
            ).execute()
            
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
            
        except HttpError as e:
            logger.error("Failed to fetch emails", error=str(e))
            return []
    
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
            self.service.users().messages().trash(
                userId="me",
                id=email_id
            ).execute()
            logger.info("Email moved to trash", email_id=email_id)
            return True
        except HttpError as e:
            logger.error("Failed to trash email", error=str(e), email_id=email_id)
            return False
    
    def untrash_email(self, email_id: str) -> bool:
        """Restore an email from trash."""
        try:
            self.service.users().messages().untrash(
                userId="me",
                id=email_id
            ).execute()
            logger.info("Email restored from trash", email_id=email_id)
            return True
        except HttpError as e:
            logger.error("Failed to untrash email", error=str(e), email_id=email_id)
            return False
    
    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read."""
        try:
            self.service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            logger.info("Email marked as read", email_id=email_id)
            return True
        except HttpError as e:
            logger.error("Failed to mark as read", error=str(e), email_id=email_id)
            return False
    
    def add_label(self, email_id: str, label_name: str) -> bool:
        """Add a label to an email (creates label if needed)."""
        try:
            # Try to find existing label
            labels_result = self.service.users().labels().list(userId="me").execute()
            label_id = None
            
            for label in labels_result.get("labels", []):
                if label["name"].lower() == label_name.lower():
                    label_id = label["id"]
                    break
            
            # Create label if not exists
            if not label_id:
                new_label = self.service.users().labels().create(
                    userId="me",
                    body={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
                ).execute()
                label_id = new_label["id"]
                logger.info("Created new label", label_name=label_name)
            
            # Add label to email
            self.service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"addLabelIds": [label_id]}
            ).execute()
            
            logger.info("Label added to email", email_id=email_id, label=label_name)
            return True
            
        except HttpError as e:
            logger.error("Failed to add label", error=str(e), email_id=email_id)
            return False
    
    def get_user_email(self) -> str:
        """Get the authenticated user's email address."""
        try:
            profile = self.service.users().getProfile(userId="me").execute()
            return profile.get("emailAddress", "")
        except HttpError as e:
            logger.error("Failed to get user profile", error=str(e))
            return ""

