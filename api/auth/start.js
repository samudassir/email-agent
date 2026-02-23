const crypto = require("crypto");

const SCOPES = [
  "https://www.googleapis.com/auth/gmail.readonly",
  "https://www.googleapis.com/auth/gmail.modify",
].join(" ");

function encrypt(text, keyHex) {
  const key = Buffer.from(keyHex, "hex");
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
  let enc = cipher.update(text, "utf8", "hex");
  enc += cipher.final("hex");
  const tag = cipher.getAuthTag().toString("hex");
  return `${iv.toString("hex")}:${enc}:${tag}`;
}

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const clientId = process.env.GOOGLE_CLIENT_ID;
  const authSecret = process.env.AUTH_SECRET;

  if (!clientId || !authSecret) {
    return res.status(500).json({ error: "Server misconfigured: missing GOOGLE_CLIENT_ID or AUTH_SECRET" });
  }

  const { gh_token, gh_owner, gh_repo } = req.body || {};
  if (!gh_token || !gh_owner || !gh_repo) {
    return res.status(400).json({ error: "Missing gh_token, gh_owner, or gh_repo" });
  }

  const payload = JSON.stringify({ gh_token, gh_owner, gh_repo });
  const encrypted = encrypt(payload, authSecret);

  res.setHeader(
    "Set-Cookie",
    `gh_auth=${encodeURIComponent(encrypted)}; Path=/api/auth; HttpOnly; Secure; SameSite=Lax; Max-Age=600`
  );

  const redirectUri = `https://${req.headers.host}/api/auth/callback`;
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: redirectUri,
    response_type: "code",
    scope: SCOPES,
    access_type: "offline",
    prompt: "consent",
  });

  return res.status(200).json({
    url: `https://accounts.google.com/o/oauth2/v2/auth?${params}`,
  });
};
