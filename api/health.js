/**
 * GET /api/health
 */
module.exports = function handler(req, res) {
  res.status(200).json({
    status: "ok",
    service: "drone-operator-localization",
    timestamp: new Date().toISOString(),
  });
};
