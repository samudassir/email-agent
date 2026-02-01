"""
Configuration management for Email Agent.
All secrets are loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google Gemini credentials (required)
    gemini_api_key: str = Field(..., description="Google Gemini API Key")
    
    # Model settings
    classifier_model: str = Field(default="gemini-2.0-flash", description="Gemini model for email classification")
    
    # Gmail OAuth paths
    gmail_credentials_file: str = Field(default="credentials.json", description="Path to OAuth credentials")
    gmail_token_file: str = Field(default="token.json", description="Path to stored OAuth token")
    
    # Agent behavior
    batch_size: int = Field(default=10, description="Number of emails to process per run")
    dry_run: bool = Field(default=True, description="If True, don't actually trash emails")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence to auto-trash")
    
    # Safety - emails from these domains/addresses are always important
    # Use comma-separated strings in .env, e.g., WHITELIST_DOMAINS=company.com,other.com
    whitelist_domains: str = Field(
        default="",
        description="Comma-separated domains that are always marked important"
    )
    whitelist_senders: str = Field(
        default="",
        description="Comma-separated email addresses that are always marked important"
    )
    
    # Opik Observability (optional - no PII is sent)
    opik_enabled: bool = Field(default=False, description="Enable Opik observability")
    opik_api_key: str = Field(default="", description="Opik API key")
    opik_project: str = Field(default="email-agent", description="Opik project name")
    opik_workspace: str = Field(default="", description="Opik workspace name")
    
    def get_whitelist_domains(self) -> list[str]:
        """Get whitelist domains as a list."""
        if not self.whitelist_domains:
            return []
        return [d.strip() for d in self.whitelist_domains.split(",") if d.strip()]
    
    def get_whitelist_senders(self) -> list[str]:
        """Get whitelist senders as a list."""
        if not self.whitelist_senders:
            return []
        return [s.strip() for s in self.whitelist_senders.split(",") if s.strip()]
    
    # Classification criteria (can be customized)
    important_criteria: str = Field(
        default="""Important emails include:
- Direct messages from colleagues, managers, or clients
- Meeting invitations for FUTURE events only
- Urgent requests or time-sensitive matters
- Financial statements or bills that require action
- Security alerts from known services
- Personal messages from friends/family
- Property management or landlord/tenant communications""",
        description="Criteria for important emails"
    )
    
    not_important_criteria: str = Field(
        default="""Not important emails include:
- Marketing newsletters and promotions
- Social media notifications
- Automated system notifications
- Cold sales outreach and spam
- General announcements/broadcasts
- Subscription confirmations for services
- Promotional offers and discounts
- LinkedIn job alerts and job notifications
- Professional organization event invitations (ISACA, meetups, conferences)
- Calendar notifications for PAST events (already happened)
- Online course promotions (Coursera, DeepLearning.ai, etc.)
- Airline loyalty program statements""",
        description="Criteria for non-important emails"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()

