import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const requireBackend = args[0] === "--require-backend";
const pythonArgs = requireBackend ? args.slice(1) : args;

const root = process.cwd();
const envPython = process.env.SMARTMIX_PYTHON;
const candidates = [
  envPython,
  join(root, ".venv", "Scripts", "python.exe"),
  join(root, "venv", "Scripts", "python.exe"),
  "D:\\miniconda\\python.exe",
  "python",
].filter(Boolean);

const uniqueCandidates = [...new Set(candidates)];
const python = selectPython(uniqueCandidates, requireBackend);

if (!python) {
  const reason = requireBackend
    ? "No Python environment with SmartMix backend dependencies was found."
    : "No Python environment with pip was found.";
  console.error(reason);
  console.error("Set SMARTMIX_PYTHON to the Python executable you want SmartMix to use.");
  process.exit(1);
}

const child = spawn(python, pythonArgs, {
  cwd: root,
  env: process.env,
  stdio: "inherit",
  windowsHide: false,
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

function selectPython(candidatesToTry, needsBackend) {
  for (const candidate of candidatesToTry) {
    if (candidate.includes("\\") && !existsSync(candidate)) continue;
    if (pythonWorks(candidate, needsBackend)) return candidate;
  }
  return null;
}

function pythonWorks(candidate, needsBackend) {
  const check = needsBackend
    ? "import fastapi, uvicorn, imageio_ffmpeg, librosa, numpy, scipy, soundfile, pyloudnorm"
    : "import pip";
  const result = spawnSync(candidate, ["-c", check], {
    cwd: root,
    encoding: "utf8",
    shell: false,
    windowsHide: true,
  });
  return result.status === 0;
}
