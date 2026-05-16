export type AudioToolchainStatus = {
  rubberbandAvailable: boolean;
  demucsAvailable: boolean;
  spleeterAvailable: boolean;
  essentiaAvailable: boolean;
  pythonAvailable: boolean;
  warnings: string[];
};

export type CommandProbe = (command: string, args: string[]) => Promise<boolean>;

export async function checkAudioToolchain(probe: CommandProbe = defaultProbe): Promise<AudioToolchainStatus> {
  const [rubberbandAvailable, pythonAvailable, demucsAvailable, spleeterAvailable, essentiaAvailable] = await Promise.all([
    probe("rubberband", ["--version"]),
    probe("python", ["--version"]),
    probe("python", ["-m", "demucs", "--help"]),
    probe("python", ["-m", "spleeter", "--help"]),
    probe("python", ["-c", "import essentia; print('ok')"]),
  ]);
  const warnings: string[] = [];
  if (!rubberbandAvailable) warnings.push("Rubber Band 不可用，将退回到 fallback time-stretch，音质可能下降。");
  if (!demucsAvailable && !spleeterAvailable) warnings.push("Demucs/Spleeter 不可用，将使用 full mix 过渡，stem 级人声和低频控制受限。");
  if (!essentiaAvailable) warnings.push("Essentia 不可用，将跳过变速/变调后的二次验证。");
  if (!pythonAvailable) warnings.push("Python 不可用，外部音频工具 wrapper 无法执行。");
  return { rubberbandAvailable, demucsAvailable, spleeterAvailable, essentiaAvailable, pythonAvailable, warnings };
}

async function defaultProbe(command: string, args: string[]): Promise<boolean> {
  try {
    const { spawn } = await import("node:child_process");
    return await new Promise((resolve) => {
      const child = spawn(command, args, { stdio: "ignore", shell: process.platform === "win32" });
      const timer = setTimeout(() => {
        child.kill();
        resolve(false);
      }, 2500);
      child.on("error", () => {
        clearTimeout(timer);
        resolve(false);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        resolve(code === 0);
      });
    });
  } catch {
    return false;
  }
}
