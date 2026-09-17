/**
 * POST /api/predict
 * Vercel Serverless Function — Python predict_api.py 호출
 */

const { execFile } = require("child_process");
const path = require("path");
const { promisify } = require("util");

const execFileAsync = promisify(execFile);

function resolvePython() {
  return process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
}

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS,POST");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version"
  );

  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  if (req.method !== "POST") {
    return res.status(405).json({ success: false, error: "Method not allowed" });
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
    const {
      latitude,
      longitude,
      bearing_degrees,
      drone_type = "unknown",
      signal_strength_dbm = -65,
    } = body;

    if (latitude === undefined || longitude === undefined || bearing_degrees === undefined) {
      return res.status(400).json({
        success: false,
        error: "MISSING_PARAMETERS",
        message: "Required: latitude, longitude, bearing_degrees",
      });
    }

    if (bearing_degrees < 0 || bearing_degrees > 359) {
      return res.status(400).json({
        success: false,
        error: "INVALID_BEARING",
        message: "bearing_degrees must be 0-359",
      });
    }

    const scriptPath = path.join(process.cwd(), "src", "predict_api.py");
    const python = resolvePython();
    const args = [
      scriptPath,
      String(latitude),
      String(longitude),
      String(bearing_degrees),
      String(drone_type),
      String(signal_strength_dbm),
    ];

    const { stdout, stderr } = await execFileAsync(python, args, {
        timeout: 60000,
      maxBuffer: 10 * 1024 * 1024,
      cwd: process.cwd(),
      env: process.env,
    });

    if (stderr) {
      console.error("Python stderr:", stderr);
    }

    const result = JSON.parse(stdout.trim().split("\n").pop());
    return res.status(result.success === false ? 400 : 200).json(result);
  } catch (error) {
    console.error("API Error:", error);
    return res.status(500).json({
      success: false,
      error: "PROCESSING_ERROR",
      message: error.message,
    });
  }
};
