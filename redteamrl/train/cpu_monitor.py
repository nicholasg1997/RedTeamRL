"""Per-process CPU sampling for training containers -- diagnoses single-thread bottlenecks for free.

Why this exists (2026-10-05): the LoRA attacker engine decoded at ~3 tok/s per sequence in training,
but ~22 tok/s with the same server settings alone and ~10 when co-located with the defender engine.
8 CPUs instead of 1 changed nothing, and the gap did not track defender load. The leading remaining
suspect is a SINGLE-THREADED process pinned at ~100% of one core (each vLLM engine's scheduler loop
and API server is one Python process; the rollout harness is one GIL-bound process). Total container
CPU cannot show that -- 4 of 8 cores busy looks healthy even if one process is saturated. So this logs
CPU per process: any line at ~100% names the bottleneck.
"""
from __future__ import annotations

import re
import threading


def label_process(cmdline: list[str], is_self: bool) -> str:
    """Human label for a process, from its command line. Pure, so it is unit-testable."""
    if is_self:
        return "trainer+rollout harness"
    joined = " ".join(cmdline)
    port = re.search(r"--port[ =](\d+)", joined)
    if "vllm" in joined and "serve" in joined:
        return f"vllm api :{port.group(1)}" if port else "vllm api"
    if "EngineCore" in joined or "vllm" in joined:
        return "vllm engine"
    if cmdline and cmdline[0].rsplit("/", 1)[-1] in ("bash", "sh"):
        return "sandbox shells"
    return (cmdline[0].rsplit("/", 1)[-1] if cmdline else "?")[:24]


def _engine_port(proc, default: str) -> str:
    """vLLM's EngineCore is a child of its API server; label it with the parent's port."""
    try:
        for parent in proc.parents():
            port = re.search(r"--port[ =](\d+)", " ".join(parent.cmdline()))
            if port:
                return f"vllm engine :{port.group(1)}"
    except Exception:
        pass
    return default


def start_cpu_monitor(interval_s: float = 60.0, top: int = 6) -> threading.Event:
    """Log the busiest processes (CPU% of ONE core; 100% = a saturated core) every interval.
    Returns an Event; set it to stop. A daemon thread, so it never blocks shutdown."""
    import psutil

    stop = threading.Event()
    me = psutil.Process()

    def loop():
        primed: dict[int, psutil.Process] = {}
        while not stop.wait(interval_s):
            try:
                procs = [me, *me.children(recursive=True)]
                totals: dict[str, float] = {}
                for p in procs:
                    if p.pid not in primed:
                        primed[p.pid] = p
                        p.cpu_percent(None)      # first call primes; it always returns 0.0
                        continue
                    try:
                        pct = primed[p.pid].cpu_percent(None)
                        label = label_process(p.cmdline(), p.pid == me.pid)
                        if label == "vllm engine":
                            label = _engine_port(p, label)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    totals[label] = totals.get(label, 0.0) + pct
                busiest = sorted(totals.items(), key=lambda kv: -kv[1])[:top]
                cores = psutil.cpu_percent(None, percpu=True)
                print("[cpu] " + "  ".join(f"{name}={pct:.0f}%" for name, pct in busiest)
                      + f"  | cores>90%: {sum(c > 90 for c in cores)}/{len(cores)}", flush=True)
            except Exception as exc:          # diagnostics must never take training down
                print(f"[cpu] monitor error: {exc!r}", flush=True)

    threading.Thread(target=loop, name="cpu-monitor", daemon=True).start()
    return stop
