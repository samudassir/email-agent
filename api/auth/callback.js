const crypto = require("crypto");
const _sodium = require("libsodium-wrappers-sumo");

function decrypt(encryptedText, keyHex) {
  const key = Buffer.from(keyHex, "hex");
  const [ivHex, enc, tagHex] = encryptedText.split(":");
  const decipher = crypto.createDecipheriv(
    "aes-256-gcm",
    key,
    Buffer.from(ivHex, "hex")
  );
  decipher.setAuthTag(Buffer.from(tagHex, "hex"));
  let dec = decipher.update(enc, "hex", "utf8");
  dec += decipher.final("utf8");
  return dec;
}

function parseCookies(header) {
  const cookies = {};
  if (!header) return cookies;
  for (const part of header.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k) cookies[k.trim()] = decodeURIComponent(v.join("="));
  }
  return cookies;
}

async function encryptForGitHub(secret, publicKeyB64) {
  await _sodium.ready;
  const sodium = _sodium;
  const binKey = sodium.from_base64(
    publicKeyB64,
    sodium.base64_variants.ORIGINAL
  );
  const binSecret = sodium.from_string(secret);
  const encrypted = sodium.crypto_box_seal(binSecret, binKey);
  return sodium.to_base64(encrypted, sodium.base64_variants.ORIGINAL);
}

module.exports = async (req, res) => {
  const { code, error: oauthError } = req.query;

  if (oauthError) {
    return res.status(400).send(errorPage(`Google OAuth error: ${oauthError}`));
  }
  if (!code) {
    return res.status(400).send(errorPage("Missing authorization code."));
  }

  const authSecret = process.env.AUTH_SECRET;
  const clientId = process.env.GOOGLE_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET;

  if (!authSecret || !clientId || !clientSecret) {
    return res.status(500).send(errorPage("Server misconfigured."));
  }

  // Read GitHub info from encrypted cookie
  const cookies = parseCookies(req.headers.cookie);
  if (!cookies.gh_auth) {
    return res
      .status(400)
      .send(
        errorPage(
          "Session expired. Please go back and try re-authenticating again."
        )
      );
  }

  let ghInfo;
  try {
    ghInfo = JSON.parse(decrypt(cookies.gh_auth, authSecret));
  } catch {
    return res
      .status(400)
      .send(errorPage("Invalid session. Please try again."));
  }

  const { gh_token, gh_owner, gh_repo } = ghInfo;

  // Exchange auth code for tokens
  const redirectUri = `https://${req.headers.host}/api/auth/callback`;
  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
      grant_type: "authorization_code",
    }),
  });

  const tokens = await tokenRes.json();
  if (tokens.error) {
    return res
      .status(400)
      .send(
        errorPage(
          `Token exchange failed: ${tokens.error_description || tokens.error}`
        )
      );
  }

  // Build token.json matching google-auth-oauthlib format
  // Omit `scopes` — google-auth sends them in refresh requests and Google
  // rejects them with invalid_scope for Web-type clients. Without the field,
  // refresh works and returns the originally-granted scopes automatically.
  const tokenJson = JSON.stringify({
    token: tokens.access_token,
    refresh_token: tokens.refresh_token,
    token_uri: "https://oauth2.googleapis.com/token",
    client_id: clientId,
    client_secret: clientSecret,
    universe_domain: "googleapis.com",
    account: "",
    expiry: new Date(Date.now() + tokens.expires_in * 1000)
      .toISOString()
      .replace("Z", "000"),
  });

  // Update GitHub secret
  try {
    const ghHeaders = {
      Authorization: `Bearer ${gh_token}`,
      Accept: "application/vnd.github.v3+json",
      "Content-Type": "application/json",
    };

    // Get repo public key
    const keyRes = await fetch(
      `https://api.github.com/repos/${gh_owner}/${gh_repo}/actions/secrets/public-key`,
      { headers: ghHeaders }
    );
    if (!keyRes.ok) {
      const body = await keyRes.text();
      throw new Error(`Failed to get repo public key (${keyRes.status}): ${body}`);
    }
    const { key, key_id } = await keyRes.json();

    // Encrypt with libsodium sealed box
    const encryptedValue = await encryptForGitHub(tokenJson, key);

    // Update the secret
    const updateRes = await fetch(
      `https://api.github.com/repos/${gh_owner}/${gh_repo}/actions/secrets/GOOGLE_TOKEN_JSON`,
      {
        method: "PUT",
        headers: ghHeaders,
        body: JSON.stringify({ encrypted_value: encryptedValue, key_id }),
      }
    );

    if (updateRes.status !== 204 && updateRes.status !== 201) {
      const body = await updateRes.text();
      throw new Error(`GitHub API returned ${updateRes.status}: ${body}`);
    }
  } catch (err) {
    return res
      .status(500)
      .send(errorPage(`Failed to update GitHub secret: ${err.message}`));
  }

  // Clear cookie
  res.setHeader(
    "Set-Cookie",
    "gh_auth=; Path=/api/auth; HttpOnly; Secure; SameSite=Lax; Max-Age=0"
  );

  res.setHeader("Content-Type", "text/html");
  return res.send(successPage());
};

function successPage() {
  return page(
    "Token Refreshed",
    `<div class="icon">&#10003;</div>
     <h2>Gmail Token Refreshed</h2>
     <p>The new token has been saved to your <code>GOOGLE_TOKEN_JSON</code> GitHub secret.</p>
     <p>Your next scheduled workflow run will use the new token automatically.</p>
     <a href="/" class="btn">Back to Email Agent</a>`
  );
}

function errorPage(message) {
  return page(
    "Error",
    `<div class="icon error">!</div>
     <h2>Something went wrong</h2>
     <p class="error-msg">${message}</p>
     <a href="/" class="btn">Back to Email Agent</a>`
  );
}

function page(title, body) {
  return `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Agent - ${title}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
:root { --bg:#0a0a0f; --card:#12121a; --accent:#6366f1; --text:#e4e4e7; --muted:#a1a1aa; --border:#27272a; --green:#22c55e; --red:#ef4444; }
body { font-family:'Space Grotesk',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; display:flex; align-items:center; justify-content:center;
  background-image:radial-gradient(ellipse at top,rgba(99,102,241,.1) 0%,transparent 50%); }
.card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:3rem; text-align:center; max-width:500px; width:90%; }
.icon { font-size:3rem; width:70px; height:70px; line-height:70px; border-radius:50%; margin:0 auto 1.5rem; background:rgba(34,197,94,.15); color:var(--green); }
.icon.error { background:rgba(239,68,68,.15); color:var(--red); }
h2 { margin-bottom:1rem; }
p { color:var(--muted); margin-bottom:1rem; line-height:1.6; }
code { background:var(--bg); padding:.2rem .5rem; border-radius:4px; font-size:.9rem; }
.error-msg { color:var(--red); background:rgba(239,68,68,.1); padding:1rem; border-radius:8px; font-size:.9rem; word-break:break-word; }
.btn { display:inline-block; margin-top:1.5rem; padding:.75rem 2rem; background:linear-gradient(135deg,var(--accent),#8b5cf6); color:#fff; border:none; border-radius:8px; font-weight:600; text-decoration:none; }
</style></head><body><div class="card">${body}</div></body></html>`;
}
