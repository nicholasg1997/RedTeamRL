"""Gate-0 teacher-viability probe (spec 2026-10-22 §4).

Runs a TEACHER model (via OpenRouter) as the scratchpad attacker against the tiers, no training, and
reports per-tier win / recovery / transform / refusal. T2 (the oracle literal redactor) is the gating
tier and needs NO GPU -- just the teacher, the pure-Python oracle, and the Modal sandbox. T0/T1 would
add the 4B defender on vLLM (a later GPU variant); this probe answers the go/no-go on T2.

    modal run --detach scripts/gate0_probe.py --teacher "z-ai/glm-5.3" --episodes 3 \
        --env-seed-start 7100 --tag glm53-prereqs-smoke

Use identical options to resume. --retry-failed retries exception-failed episodes only.
Legacy tags without a manifest and tags from changed code/config are rejected.
"""
import modal

RUN_ROOT = "/runs/gate0"

image = (
    modal.Image.from_registry("python:3.12-slim")
    .pip_install("requests", "pydantic", "pyyaml")
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-gate0", image=image)


@app.function(cpu=4.0, timeout=60 * 60, secrets=[modal.Secret.from_name("openrouter")],
              volumes={"/runs": runs})
def probe(teacher: str, episodes: int = 15, tier: str = "T2", tag: str = "glm53-v2",
          env_seed_start: int = 7000, workers: int = 4, max_turns: int = 12,
          max_tokens: int = 8192, retry_failed: bool = False):
    import os
    from pathlib import Path
    import requests
    from redteamrl.policies.openrouter_client import make_openrouter_generate
    from redteamrl.sandbox.modal_shell import ModalShellSandbox
    from redteamrl.scratchpad.collection import probe_config, execute_teacher_episode, run_collection
    from redteamrl.scratchpad.experiment import source_digest

    config = probe_config(teacher=teacher, episodes=episodes, tier=tier, tag=tag,
                          source=source_digest("/root/redteamrl", __file__),
                          env_seed_start=env_seed_start, workers=workers,
                          max_turns=max_turns, max_tokens=max_tokens)
    runs.reload()
    key = os.environ["OPENROUTER_API_KEY"]

    def execute(job):
        # No transport, callback buffer or generation counter is shared between episodes.
        with requests.Session() as session:
            def generate(on_response):
                return make_openrouter_generate(teacher, key, session=session,
                    max_tokens=config["max_tokens"], temperature=config["temperature"],
                    top_p=config["top_p"], timeout=config["api"]["timeout_s"],
                    max_retries=config["api"]["max_retries"], backoff_s=config["api"]["backoff_s"],
                    on_response=on_response)
            return execute_teacher_episode(job, config, generate, lambda: ModalShellSandbox(
                app, image, **config["sandbox"]))

    report = run_collection(config, Path(RUN_ROOT) / tag, execute,
                            commit=runs.commit, retry_failed=retry_failed)
    summary = report["summary"]
    row = summary[tier]
    flags = row["diagnostics"]["flags"]
    win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "n/a"
    print(f"\nGATE 0 {teacher} {tier}: completed={row['n_completed']}/{row['n_requested']} "
          f"failed={row['n_failed']} win={win} "
          f"recovery={row['recovery_wins']}/{row['recovery_eligible']} "
          f"invalid={flags.get('invalid_action', 0)}/{row['diagnostics']['turns']} "
          f"empty={flags.get('empty_response', 0)} actionless={flags.get('actionless_response', 0)} "
          f"execution_errors={flags.get('execution_error', 0)} "
          f"execution_error_hints={flags.get('execution_error_text_hint', 0)} "
          f"incorrect_submits={flags.get('incorrect_submit', 0)} -> {summary['gate_status']}", flush=True)
    return report


@app.local_entrypoint()
def main(teacher: str = "z-ai/glm-5.3", episodes: int = 15, tier: str = "T2", tag: str = "glm53-v2",
         env_seed_start: int = 7000, workers: int = 4, max_turns: int = 12,
         max_tokens: int = 8192, retry_failed: bool = False):
    from redteamrl.scratchpad.collection import probe_config
    probe_config(teacher=teacher, episodes=episodes, tier=tier, tag=tag, source="preflight",
                 env_seed_start=env_seed_start, workers=workers, max_turns=max_turns, max_tokens=max_tokens)
    probe.remote(teacher, episodes, tier, tag, env_seed_start, workers, max_turns, max_tokens, retry_failed)
