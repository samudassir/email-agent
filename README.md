# Email Agent 🤖📧

An autonomous AI agent that reads your unread Gmail emails, classifies them by importance using Google Gemini, and automatically trashes non-important ones.

## Features

- 🔍 **Smart Classification** — Uses Google Gemini to understand email context and intent
- 🛡️ **Safety First** — Dry-run mode (default), undo functionality, and whitelist support
- ⚡ **Autonomous** — Runs independently, makes decisions, takes actions
- 📊 **Transparent** — Shows reasoning and confidence for every classification
- 🎯 **Customizable** — Define your own criteria for important vs. not important
- 📜 **Audit Trail** — Logs all actions for review and accountability

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Gmail API     │────▶│   Classifier    │────▶│   Decision      │
│   (Unread)      │     │   (Gemini)      │     │   Engine        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                               │                               │
                        ▼                               ▼                               ▼
                ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
                │   IMPORTANT   │              │  NOT IMPORTANT │              │   UNCERTAIN   │
                │   → Keep      │              │   → Trash      │              │   → Review    │
                └───────────────┘              └───────────────┘              └───────────────┘
```

## Prerequisites

- Python 3.10+
- Google Cloud account (for Gmail API OAuth)
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

## Quick Start

```bash
cd email-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure (see Setup section below)
# Then authenticate with Gmail
python agent.py auth

# Run in dry-run mode (safe preview)
python agent.py run

# Run for real
python agent.py run --no-dry-run
```

## Setup

### 1. Get Gemini API Key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **Create API Key**
3. Copy the key

### 2. Set Up Gmail API (Google Cloud Console)

#### Create Project & Enable API

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create a new project (e.g., "Email Agent")
3. Go to **APIs & Services** → **Library**
4. Search for **Gmail API** → Click **Enable**

#### Configure OAuth Consent Screen

1. Go to **APIs & Services** → **OAuth consent screen**
2. Fill in basic info:
   - App name: `Email Agent`
   - User support email: your email
   - Developer contact: your email
3. Add yourself as a **Test User**
4. Save

#### Create OAuth Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **OAuth client ID**
3. Application type: **Desktop app**
4. Name: `Email Agent`
5. Click **Create**
6. Download JSON or copy Client ID & Secret

#### Create credentials.json

Create `credentials.json` in the project folder:

```json
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "client_secret": "YOUR_CLIENT_SECRET",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "redirect_uris": ["http://localhost"]
  }
}
```

### 3. Create .env File

Create `.env` in the project folder:

```bash
# Google Gemini API Key (Required)
GEMINI_API_KEY=your-gemini-api-key-here

# Agent Behavior
DRY_RUN=true
CONFIDENCE_THRESHOLD=0.8
BATCH_SIZE=10

# Whitelist - emails from these are ALWAYS kept (comma-separated)
WHITELIST_DOMAINS=yourcompany.com,importantnewsletter.com
WHITELIST_SENDERS=
```

### 4. Authenticate with Gmail

```bash
python agent.py auth
```

A browser will open for Google OAuth. Sign in and authorize the app.

## Usage

### Preview Mode (Dry Run)

```bash
python agent.py run
```

Shows what would happen without actually trashing emails.

### Process Emails for Real

```bash
python agent.py run --no-dry-run
```

### Process Specific Number of Emails

```bash
python agent.py run --batch 20 --no-dry-run
```

### Undo Last Trash

```bash
python agent.py undo
```

### View Action History

```bash
python agent.py history
```

## Example Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🤖 Email Agent                                                               ║
║ Autonomous email classification and organization                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
Logged in as: user@gmail.com

📬 Fetching unread emails...
Found 5 unread email(s)

🤖 Classifying emails...

                        📧 Email Classification Results                         
╭─────────────┬─────────────┬──────────────┬─────────────┬────────────┬────────╮
│ From        │ Subject     │ Category     │ Importance  │ Confidence │ Action │
├─────────────┼─────────────┼──────────────┼─────────────┼────────────┼────────┤
│ Boss        │ Q4 Review   │ work         │ ✓ Important │        95% │ KEEP   │
│ Corp Bank   │ Your stmt   │ financial    │ ✓ Important │        90% │ KEEP   │
│ NewsletterX │ Weekly tips │ whitelisted  │ ✓ Important │       100% │ KEEP   │
│ Deals.com   │ 50% off!    │ promotional  │ ✗ Not       │        90% │ TRASH  │
│             │             │              │ Important   │            │        │
│ SocialApp   │ New likes   │ notification │ ✗ Not       │        85% │ TRASH  │
│             │             │              │ Important   │            │        │
╰─────────────┴─────────────┴──────────────┴─────────────┴────────────┴────────╯

  🗑️ Trashed: 50% off!...
  🗑️ Trashed: New likes...

╭───────────────────────────────── 📊 Summary ─────────────────────────────────╮
│ 📧 Processed: 5                                                              │
│ ✅ Important: 3                                                              │
│ ❌ Not Important: 2                                                          │
│ 🗑️  Trashed: 2                                                               │
│ 📥 Kept: 3                                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `CLASSIFIER_MODEL` | `gemini-2.0-flash` | Gemini model for classification |
| `BATCH_SIZE` | `10` | Number of emails to process per run |
| `DRY_RUN` | `true` | If true, preview only (no actual trashing) |
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum confidence (0-1) to auto-trash |
| `WHITELIST_DOMAINS` | `` | Comma-separated domains always marked important |
| `WHITELIST_SENDERS` | `` | Comma-separated emails always marked important |

### Whitelist

Emails from whitelisted domains/senders bypass AI classification entirely and are always marked as **Important (100% confidence)**.

```bash
WHITELIST_DOMAINS=yourcompany.com,trustedsite.com
WHITELIST_SENDERS=boss@example.com,important@partner.com
```

### Classification Criteria

The default criteria classifies as **Not Important**:
- Marketing newsletters and promotions
- Job alert notifications
- Social media notifications
- Professional event invitations and meetups
- Calendar notifications for past events
- Online course promotions
- Loyalty program statements

Classified as **Important**:
- Direct messages from colleagues/clients
- Future meeting invitations
- Financial bills requiring action
- Security alerts
- Personal messages

You can customize these in `config.py`.

## Project Structure

```
email-agent/
├── agent.py             # Main CLI application
├── gmail_client.py      # Gmail API wrapper
├── classifier.py        # Gemini-based email classifier
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── credentials.json     # Gmail OAuth credentials (you create)
├── token.json           # OAuth token (auto-created)
├── .env                 # Your API keys and settings (you create)
├── agent_actions.json   # Action log (auto-created)
└── README.md
```

## Free Tier Limits

Gemini free tier has **~20 requests per day per model**. To maximize:

1. Use whitelist for known-important senders (no API call needed)
2. Run with smaller batches: `--batch 10`
3. Quota resets daily at midnight Pacific time

For unlimited usage, enable billing at [aistudio.google.com](https://aistudio.google.com) (pay-as-you-go is very cheap).

## Safety Features

| Feature | Description |
|---------|-------------|
| **Dry Run Mode** | Default mode - shows preview without taking action |
| **Confidence Threshold** | Only auto-trashes when 80%+ confident |
| **Whitelist** | Specified domains/senders always kept |
| **Undo** | Restore the last trashed email |
| **Action Log** | All actions logged with timestamps |
| **Trash (not Delete)** | Emails go to Trash, recoverable for 30 days |

## Why This Is a True AI Agent

Unlike simple chatbots, this is a genuine **autonomous agent**:

| Agent Trait | Implementation |
|-------------|----------------|
| **Perceives Environment** | Reads emails from Gmail API |
| **Reasons** | LLM analyzes content, sender, context |
| **Makes Decisions** | Classifies as important/not important |
| **Takes Actions** | Moves emails to trash autonomously |
| **Has Goals** | Keep inbox clean and organized |
| **Maintains State** | Logs actions, supports undo |

## Troubleshooting

### "credentials.json not found"
Create the file manually (see Setup section) or download from Google Cloud Console.

### "Token has been expired or revoked"
```bash
rm token.json
python agent.py auth
```

### Rate limit errors (429)
You've hit the free tier daily limit. Wait until tomorrow or enable billing.

### Classification seems wrong
1. Add the sender to whitelist
2. Adjust criteria in `config.py`
3. Increase `CONFIDENCE_THRESHOLD` for more conservative behavior

## Security Notes

- OAuth tokens stored locally in `token.json`
- Never commit `credentials.json`, `token.json`, or `.env` to git
- Email content is sent to Google Gemini for classification
- All secrets loaded from environment variables
- Emails are trashed (recoverable), not permanently deleted

## License

MIT License
