from pathlib import Path
import os, subprocess, sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
WAVE = SRC / "元旦磁暴_跨层波数1"
REGION = WAVE / "two_region_mechanism_profiles_20261021"
ERA = SRC / "A研究进展报告" / "ERA5_2025元旦磁暴研究报告"
SCRIPTS = {
    "2": REGION / "plot_figure2_temperature_dailymean_20261121.py",
    "3": WAVE / "plot_combined_environment_w1_ep_qy_legend_bottom_20261109.py",
    "4": ERA / "plot_figure4_phase_envelope_180mm_aligned_20261107.py",
    "5": ERA / "plot_figure5_height_wind_weather_180mm_20261108.py",
}
if len(sys.argv) != 2 or sys.argv[1] not in (*SCRIPTS, "all"):
    raise SystemExit("Usage: python run_figures.py 2|3|4|5|all")
env = os.environ.copy()
paths = [str(WAVE), str(REGION), str(ERA)]
if env.get("PYTHONPATH"):
    paths.append(env["PYTHONPATH"])
env["PYTHONPATH"] = os.pathsep.join(paths)
choices = SCRIPTS if sys.argv[1] == "all" else [sys.argv[1]]
for number in choices:
    print(f"Running Figure {number}...", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS[number])], cwd=SRC, env=env, check=True)
