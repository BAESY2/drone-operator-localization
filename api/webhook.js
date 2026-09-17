/**
 * Telegram webhook (optional) — forwards updates to predict API.
 * Prefer long-polling bot (telegram/bot.py) for MVP.
 */
module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, error: "Method not allowed" });
  }

  // Placeholder: wire to Bot API when using webhook mode
  return res.status(200).json({
    ok: true,
    message: "Webhook received. Use telegram/bot.py polling for MVP.",
  });
};
