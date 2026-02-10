# Email Agent 🤖📧

An autonomous AI agent that keeps your inbox clean — saving **4 hours/month** on email cleanup and **$120/year** on cloud storage. It reads your unread Gmail, classifies each email by importance using Google Gemini, and automatically trashes the noise. It learns from your corrections to get smarter over time.

## The Problem

The average knowledge worker spends **2.5 hours/day** on email. Most of that is wading through newsletters, promotions, social notifications, and automated alerts that don't need attention. Meanwhile, Gmail's 15 GB fills up with years of junk, forcing you to pay for extra storage or manually clean up.

## The Solution

Email Agent is an **autonomous AI agent** that does the triage for you:

1. **Perceives** — Connects to your Gmail and reads unread emails
2. **Reasons** — Uses Google Gemini to analyze sender, content, and context
3. **Decides** — Classifies each email as important, not important, or uncertain
4. **Acts** — Auto-trashes low-value emails (newsletters, promos, old notifications)
5. **Learns** — Records corrections when you undo a mistake, and feeds those patterns back into future classifications

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Gmail API   │────▶│  Gemini LLM      │────▶│  Decision Engine │────▶│  Take Action │
│  (Fetch)     │     │  (Classify)      │     │  (Confidence +   │     │  (Trash/Keep)│
│              │     │                  │     │   Age + Context)  │     │              │
└──────────────┘     └──────────────────┘     └──────────────────┘     └──────┬───────┘
                                                                              │
                     ┌──────────────────┐     ┌──────────────────┐            │
                     │  Context Store   │◀────│  User Correction │◀───────────┘
                     │  (Learn)         │     │  (Undo → Learn)  │    Feedback Loop
                     └──────────────────┘     └──────────────────┘
```

## Why This Is a True AI Agent

| Agent Trait | Implementation |
|---|---|
| **Perceives environment** | Reads emails via Gmail API (sender, subject, body, date, labels) |
| **Reasons about context** | LLM classifies with structured reasoning; sender history injected into prompt |
| **Makes autonomous decisions** | Confidence threshold + age-based rules + whitelist logic |
| **Takes real actions** | Trashes emails via Gmail API (recoverable for 30 days) |
| **Learns from feedback** | Undo triggers correction recording → future prompts adjusted |
| **Maintains state** | Context store tracks 17+ sender domains across sessions |

## Use of LLMs

- **Google Gemini 2.0 Flash** for classification via structured JSON output
- **Batch classification** — up to 10 emails per API call for efficiency
- **Security-hardened system prompt** with `<EMAIL>` boundary tags and explicit injection defenses
- **Sender history context** injected into prompt — the LLM sees historical classification patterns per domain
- **Correction patterns** automatically added to the system prompt when the agent has been wrong before
- **Output guardrails** validate and auto-fix every LLM response (7-step pipeline: JSON → fields → importance → confidence → category → action → reasoning)
- **Age-based post-processing** overrides LLM decisions for old emails (90d+ and 1y+ thresholds with protected categories)

## Evaluation & Observability

This is where the project goes deep. Four layers of evaluation:

### 1. Opik Integration (Online Observability)

Every run produces a **full session trace** with nested spans, all PII-scrubbed before leaving the machine:

```
email_processing_session (trace)
├── fetch_emails (span) — latency, email count
├── llm_call (span) — model, tokens, latency, response, error type
├── classify_email (span) — per-email: domain, importance, confidence, category
├── email_action_trash (span) — action taken, success, dry_run flag
├── suspicious_activity (span) — prompt injection detection with risk level
└── session_complete (span) — aggregate metrics
```

**Online feedback scores** logged on every trace:
- `json_parse_success` — did the LLM return valid JSON?
- `response_completeness` — batch returned all expected classifications?
- `classification_confidence` — average confidence
- `error_free_rate` — fraction processed without errors
- `high_confidence_rate` — fraction with confidence ≥ 0.5
- `decisive_rate` — fraction classified as important/not_important (vs uncertain)
- `session_quality` — composite weighted score (40% confidence, 25% error-free, 20% high-confidence, 15% decisive)

### 2. Prompt Optimization (Opik Experiments)

`optimize_prompt.py` uses Opik's `evaluate()` framework to compare prompt variants:

- **19 test cases** stored as an Opik dataset (promotional, newsletters, job alerts, work, financial, security, personal, edge cases)
- **4 scoring functions**: importance accuracy, JSON validity, confidence calibration, category match
- **2 prompt variants** (baseline vs v2-structured) with side-by-side comparison
- **Optimization tracking** via `opik.create_optimization()` — results visible in the Opik dashboard

### 3. Security Tests (Garak-Style)

`evals/security_tests.py` runs **12 adversarial tests** across 4 categories:

| Category | Tests | What It Checks |
|---|---|---|
| Prompt Injection | 5 | Basic injection, system override, JSON injection, delimiter confusion, instruction smuggling |
| Jailbreak | 3 | DAN, roleplay, hypothetical scenario |
| Data Exfiltration | 2 | System prompt leak, instruction reveal |
| Classification Manipulation | 2 | CEO impersonation, keyword stuffing |

### 4. Promptfoo (Classification Accuracy)

`evals/promptfoo.yaml` — 19 test cases with assertions on importance and confidence, runnable via `npx promptfoo eval`.

### PII Scrubbing

All observability data passes through `pii_scrubber.py` before reaching Opik:
- Email addresses → domain only + SHA-256 hash
- Subject/body → length only (content never sent)
- Timestamps → hour-of-day + day-of-week only
- Regex scrubbing for emails, phone numbers, SSNs in LLM response text

## Safety & Security

| Feature | How It Works |
|---|---|
| **Dry-run mode** | Default — shows preview without trashing anything |
| **Confidence threshold** | Only auto-trashes at 80%+ confidence |
| **Whitelist** | Specified domains/senders always marked important (bypass LLM) |
| **Undo** | Restore last trashed email + record correction for learning |
| **Protected categories** | Financial, personal, and job opportunity emails shielded from auto-trash |
| **Age-aware rules** | Emails 90d–1y: protect work/financial/personal. Emails 1y+: protect financial/personal only |
| **Prompt injection defense** | System prompt rejects injected instructions; 13 injection patterns monitored |
| **Trash, not delete** | Emails go to Gmail Trash — recoverable for 30 days |
| **Fallback API key** | Automatic failover when primary Gemini key hits quota limits |
| **Action audit log** | Every trash action logged with timestamp, sender, classification, and reasoning |

## Project Structure

```
email-agent/
├── agent.py                 # Main agent — CLI, orchestration, action loop
├── classifier.py            # Gemini-based email classifier (batch + single)
├── gmail_client.py          # Gmail API client (OAuth2, fetch, trash, undo)
├── config.py                # Settings from environment variables
├── context_store.py         # Persistent sender history + correction learning
├── guardrails_validator.py  # 7-step LLM output validation with auto-fix
├── opik_integration.py      # Observability — traces, spans, feedback scores
├── pii_scrubber.py          # PII removal before sending to Opik
├── optimize_prompt.py       # Prompt optimization via Opik evaluate()
├── run_evals.py             # Evaluation runner (security + guardrails + promptfoo)
├── evals/
│   ├── security_tests.py    # 12 adversarial security tests (Garak-style)
│   ├── promptfoo.yaml       # 19 classification accuracy tests
│   └── prompt_template.txt  # Prompt template for promptfoo
├── test_spans.py            # Opik span verification script
├── web_app.py               # Flask web UI with natural language input
└── web_ui/index.html        # Static web UI (GitHub Actions trigger)
```

## Quick Start

```bash
cd email-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure .env (see env.example)
# Authenticate with Gmail
python agent.py auth

# Dry run — safe preview
python agent.py run

# Process for real
python agent.py run --no-dry-run

# Clean up old emails
python agent.py run --batch 20 --older-than 6m --no-dry-run

# Undo last trash
python agent.py undo

# Run evaluations
python run_evals.py

# Run prompt optimization
python optimize_prompt.py --compare
```

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║ 🤖 Email Agent                                              ║
║ Autonomous email classification and organization            ║
╚══════════════════════════════════════════════════════════════╝
Logged in as: user@gmail.com

📬 Fetching unread emails...
Found 5 unread email(s)

🤖 Classifying emails...

                    📧 Email Classification Results
╭─────────────┬──────────────┬──────────────┬────────────┬────────╮
│ From        │ Subject      │ Importance   │ Confidence │ Action │
├─────────────┼──────────────┼──────────────┼────────────┼────────┤
│ Boss        │ Q4 Review    │ ✓ Important  │        95% │ KEEP   │
│ Corp Bank   │ Your stmt    │ ✓ Important  │        90% │ KEEP   │
│ Newsletter  │ Weekly tips  │ ✓ Important  │       100% │ KEEP   │
│ Deals.com   │ 50% off!     │ ✗ Not Imp.   │        90% │ TRASH  │
│ SocialApp   │ New likes    │ ✗ Not Imp.   │        85% │ TRASH  │
╰─────────────┴──────────────┴──────────────┴────────────┴────────╯

  🗑️ Trashed: 50% off!...
  🗑️ Trashed: New likes...

╭──────────────────── 📊 Summary ────────────────────╮
│ 📧 Processed: 5                                     │
│ ✅ Important: 3                                     │
│ ❌ Not Important: 2                                 │
│ 🗑️  Trashed: 2                                      │
│ 📥 Kept: 3                                          │
╰─────────────────────────────────────────────────────╯
```

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash
- **Email**: Gmail API (OAuth2)
- **Observability**: Opik (traces, spans, experiments, feedback scores)
- **Evals**: Opik evaluate(), Promptfoo, custom security suite
- **Framework**: Python 3.10+, Typer (CLI), Flask (web), Rich (terminal UI)
- **Config**: Pydantic Settings + dotenv

## License

MIT
