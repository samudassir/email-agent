"""
Simple web interface for the Email Agent.
Accepts natural language prompts to control the agent.
"""

import re
import subprocess
import sys
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #27272a;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at top, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
            padding-top: 2rem;
        }
        
        .logo {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        
        .prompt-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .prompt-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .prompt-input {
            width: 100%;
            padding: 1rem 1.25rem;
            font-size: 1.1rem;
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-tertiary);
            border: 2px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }
        
        .prompt-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 4px var(--accent-glow);
        }
        
        .prompt-input::placeholder {
            color: var(--text-secondary);
            opacity: 0.6;
        }
        
        .examples {
            margin-top: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .example-chip {
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .example-chip:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }
        
        .submit-btn {
            width: 100%;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
            border: none;
            border-radius: 12px;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px var(--accent-glow);
        }
        
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .output-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }
        
        .output-header {
            padding: 1rem 1.5rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .output-title {
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-running {
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent);
        }
        
        .status-success {
            background: rgba(34, 197, 94, 0.2);
            color: var(--success);
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.2);
            color: var(--error);
        }
        
        .output-content {
            padding: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            line-height: 1.6;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: var(--text-secondary);
        }
        
        .output-content:empty::before {
            content: "Output will appear here...";
            color: var(--text-secondary);
            opacity: 0.5;
        }
        
        .parsed-params {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        .param-item {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }
        
        .param-key {
            color: var(--accent);
        }
        
        .param-value {
            color: var(--text-primary);
        }
        
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid var(--accent);
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🤖📧</div>
            <h1>Email Agent</h1>
            <p class="subtitle">Autonomous email classification powered by AI</p>
        </header>
        
        <div class="prompt-section">
            <div class="prompt-label">What would you like to do?</div>
            <input 
                type="text" 
                class="prompt-input" 
                id="prompt" 
                placeholder="e.g., Do a dry run of unread emails older than 6 months"
                autofocus
            >
            <div class="examples">
                <span class="example-chip" onclick="setPrompt('Do a dry run with 10 emails')">Dry run (10 emails)</span>
                <span class="example-chip" onclick="setPrompt('Process emails older than 1 year')">Older than 1 year</span>
                <span class="example-chip" onclick="setPrompt('Dry run of 20 emails older than 6 months')">6 months old</span>
                <span class="example-chip" onclick="setPrompt('Clean up 50 old promotional emails')">Clean 50 emails</span>
            </div>
            <div class="parsed-params" id="parsedParams" style="display: none;">
                <strong>Parsed parameters:</strong>
                <div id="paramsList"></div>
            </div>
            <button class="submit-btn" id="submitBtn" onclick="runAgent()">
                Run Email Agent
            </button>
        </div>
        
        <div class="output-section">
            <div class="output-header">
                <span class="output-title">
                    <span>📋</span> Output
                </span>
                <span class="status-badge" id="statusBadge" style="display: none;">Ready</span>
            </div>
            <div class="output-content" id="output"></div>
        </div>
        
        <footer>
            Email Agent • Context-aware classification • Learns from corrections
        </footer>
    </div>
    
    <script>
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
            parsePrompt();
        }
        
        function parsePrompt() {
            const prompt = document.getElementById('prompt').value.toLowerCase();
            const params = {
                batch: 10,
                dryRun: true,
                olderThan: null
            };
            
            // Parse batch size
            const batchMatch = prompt.match(/(\\d+)\\s*(emails?)?/);
            if (batchMatch) {
                params.batch = parseInt(batchMatch[1]);
            }
            
            // Parse dry run
            if (prompt.includes('process') || prompt.includes('clean') || prompt.includes('trash')) {
                if (!prompt.includes('dry')) {
                    params.dryRun = false;
                }
            }
            
            // Parse older than
            const olderMatch = prompt.match(/older\\s*than\\s*(\\d+)\\s*(year|month|week|day|y|m|w|d)s?/i);
            if (olderMatch) {
                const num = olderMatch[1];
                const unit = olderMatch[2][0].toLowerCase();
                params.olderThan = num + unit;
            }
            
            // Show parsed params
            const paramsDiv = document.getElementById('parsedParams');
            const paramsList = document.getElementById('paramsList');
            
            if (prompt.trim()) {
                paramsDiv.style.display = 'block';
                paramsList.innerHTML = `
                    <div class="param-item"><span class="param-key">batch:</span> <span class="param-value">${params.batch}</span></div>
                    <div class="param-item"><span class="param-key">dry_run:</span> <span class="param-value">${params.dryRun}</span></div>
                    <div class="param-item"><span class="param-key">older_than:</span> <span class="param-value">${params.olderThan || 'none'}</span></div>
                `;
            } else {
                paramsDiv.style.display = 'none';
            }
            
            return params;
        }
        
        document.getElementById('prompt').addEventListener('input', parsePrompt);
        
        async function runAgent() {
            const params = parsePrompt();
            const btn = document.getElementById('submitBtn');
            const output = document.getElementById('output');
            const statusBadge = document.getElementById('statusBadge');
            
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running...';
            output.textContent = '';
            statusBadge.style.display = 'inline-block';
            statusBadge.className = 'status-badge status-running';
            statusBadge.textContent = 'Running';
            
            try {
                const response = await fetch('/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                
                const data = await response.json();
                output.textContent = data.output;
                
                if (data.success) {
                    statusBadge.className = 'status-badge status-success';
                    statusBadge.textContent = 'Complete';
                } else {
                    statusBadge.className = 'status-badge status-error';
                    statusBadge.textContent = 'Error';
                }
            } catch (error) {
                output.textContent = 'Error: ' + error.message;
                statusBadge.className = 'status-badge status-error';
                statusBadge.textContent = 'Error';
            }
            
            btn.disabled = false;
            btn.innerHTML = 'Run Email Agent';
        }
        
        // Allow Enter key to submit
        document.getElementById('prompt').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runAgent();
        });
    </script>
</body>
</html>
"""

def parse_prompt(prompt: str) -> dict:
    """Parse natural language prompt into agent parameters."""
    prompt_lower = prompt.lower()
    
    params = {
        "batch": 10,
        "dry_run": True,
        "older_than": None
    }
    
    # Parse batch size
    batch_match = re.search(r'(\d+)\s*(emails?)?', prompt_lower)
    if batch_match:
        params["batch"] = int(batch_match.group(1))
    
    # Parse dry run vs real
    if any(word in prompt_lower for word in ['process', 'clean', 'trash', 'delete', 'remove']):
        if 'dry' not in prompt_lower:
            params["dry_run"] = False
    
    # Parse older than
    older_match = re.search(r'older\s*than\s*(\d+)\s*(year|month|week|day|y|m|w|d)s?', prompt_lower)
    if older_match:
        num = older_match.group(1)
        unit = older_match.group(2)[0].lower()
        params["older_than"] = f"{num}{unit}"
    
    return params


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/run', methods=['POST'])
def run_agent():
    """Run the email agent with parsed parameters."""
    data = request.json
    
    # Build command
    cmd = [sys.executable, "agent.py", "run"]
    cmd.extend(["--batch", str(data.get("batch", 10))])
    
    if data.get("dry_run", True):
        pass  # dry run is default
    else:
        cmd.append("--no-dry-run")
    
    if data.get("older_than"):
        cmd.extend(["--older-than", data["older_than"]])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="."
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n\n--- STDERR ---\n" + result.stderr
        
        # Strip ANSI codes for clean display
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        output = ansi_escape.sub('', output)
        
        return jsonify({
            "success": result.returncode == 0,
            "output": output,
            "command": " ".join(cmd)
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "output": "Error: Command timed out after 120 seconds"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "output": f"Error: {str(e)}"
        })


if __name__ == '__main__':
    print("Starting Email Agent Web UI...")
    print("Open http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=True)

