"""Canonical, deterministic recompute of every QAOA-dependent portfolio result (SI Section S2).

The portfolio QAOA training (p = 1, 100 gradient steps) is sensitive to floating-point
reduction order, so its trained angles change with the number of BLAS/OpenMP threads.
This runner pins ONE thread for every numerical library, records the environment and the
SHA-256 of every code and data input, runs all experiments with seeds 0-9 (one working
directory per job, each with its own copy of baseline_enumeration.csv), and writes a
SHA-256 manifest of all outputs.

Usage (from portfolio/): python run_canonical.py OUT_DIR
"""
import hashlib, json, os, platform, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
PY = sys.executable
IPOPT_BIN = os.path.join(os.path.dirname(PY), "Library", "bin")
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS"]

JOBS = {
    "fase2_attractor_matrix_convergence": ["Fase2.py"],
    "hamiltonian_ablation": ["Fase3_separacion_efectos.py", "--exp", "1", "--seeds", "10", "--bmax", "20"],
    "capped_budget": ["Fase3_separacion_efectos.py", "--exp", "3", "--seeds", "10", "--bmax", "20"],
    "frequency_policy": ["Fase3_separacion_efectos.py", "--exp", "4", "--seeds", "10", "--bmax", "20"],
    "frozen_distribution": ["Fase3_separacion_efectos.py", "--exp", "5", "--seeds", "10", "--bmax", "20"],
    "depth_and_uniform": ["recompute_s11_depth.py"],
    "mixer_A": ["recompute_mixer_ext.py", "A", "ring:by_pairs", "line:by_pairs"],
    "mixer_B": ["recompute_mixer_ext.py", "B", "fully_connected:by_pairs", "fully_connected:none"],
    "mixer_C": ["recompute_mixer_ext.py", "C", "line:none"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_record():
    import importlib.metadata as md
    pk = {}
    for p in ["numpy", "scipy", "pandas", "pennylane", "pennylane_lightning", "autograd", "pyomo", "matplotlib"]:
        try:
            pk[p] = md.version(p)
        except Exception:
            pk[p] = "n/a"
    try:
        ip = subprocess.run([os.path.join(IPOPT_BIN, "ipopt.exe"), "--version"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        ip = f"n/a ({e})"
    return {"python": sys.version, "platform": platform.platform(), "packages": pk, "ipopt": ip,
            "thread_env": {v: "1" for v in THREAD_VARS}, "seeds": list(range(10))}


def main(out):
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update({v: "1" for v in THREAD_VARS})
    env["PATH"] = IPOPT_BIN + os.pathsep + env["PATH"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    inputs = {f: sha256(os.path.join(CODE, f)) for f in sorted(os.listdir(CODE)) if os.path.isfile(os.path.join(CODE, f))}
    inputs["run_canonical.py"] = sha256(os.path.abspath(__file__))
    json.dump({"environment": env_record(), "input_sha256": inputs, "jobs": JOBS,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(out, "run_record.json"), "w"), indent=2)
    procs = {}
    for name, argv in JOBS.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(CODE, "baseline_enumeration.csv"), d)
        log = open(os.path.join(d, "run.log"), "w", encoding="utf-8")
        procs[name] = (subprocess.Popen([PY, os.path.join(CODE, argv[0])] + argv[1:], cwd=d, env=env,
                                        stdout=log, stderr=subprocess.STDOUT), log, time.time())
        print(f"started {name}", flush=True)
    status = {}
    while procs:
        for name in list(procs):
            p, log, t0 = procs[name]
            if p.poll() is not None:
                log.close()
                status[name] = {"returncode": p.returncode, "seconds": round(time.time() - t0, 1)}
                print(f"finished {name}: rc={p.returncode} in {status[name]['seconds']} s", flush=True)
                del procs[name]
        time.sleep(5)
    manifest = {}
    for root, _, files in os.walk(out):
        for f in files:
            if f in ("MANIFEST_SHA256.json",):
                continue
            pth = os.path.join(root, f)
            manifest[os.path.relpath(pth, out).replace("\\", "/")] = sha256(pth)
    json.dump({"job_status": status, "files": dict(sorted(manifest.items()))},
              open(os.path.join(out, "MANIFEST_SHA256.json"), "w"), indent=2)
    print("ALL DONE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
