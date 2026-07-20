"""Defender GRPO de-risk. Trains Qwen3-1.7B + LoRA to fix label-vs-value redaction.
Spec: docs/specs/2026-07-16-defender-grpo-derisk-design.md (LOCAL ONLY).
Run: modal run --detach scripts/train_defender.py
"""
import modal

DEFENDER_MODEL = "Qwen/Qwen3-4B"
ATTACKER_MODEL = "Qwen/Qwen3-8B"
N_ROLLOUTS = 4 #8
N_ITERS = 15 #40
LR = 1e-5
TEMP = 0.7

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch>=2.2", "transformers<5", "peft>=0.11", "accelerate>=0.30",
                 "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-defender-grpo", image=image)


@app.function(gpu="A10", timeout=24 * 60 * 60,
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def train():
    import glob, os, sys, torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from redteamrl.tasks.schema import load_task_spec
    from redteamrl.policies.prompted import PromptedAttacker, PromptedDefender
    from redteamrl.sandbox.local import LocalSandbox
    from redteamrl.train.capture import CapturingGenerate
    from redteamrl.train.learner import Learner
    from redteamrl.train.train import rollout, assign_advantages, update_step
    from redteamrl.train.eval_split import split_tasks, benign_metrics, HELD_OUT, scenario_of

    tok = AutoTokenizer.from_pretrained(DEFENDER_MODEL)
    base = AutoModelForCausalLM.from_pretrained(DEFENDER_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    learner = Learner(model, tok, opt, reward_key="defender_reward")

    cap = CapturingGenerate(model, tok, temperature=TEMP)
    def defender_factory():
        return PromptedDefender(generate=cap)          # defender uses the capturing generate

    v1 = "/root/redteamrl/data/tasks/v1"
    all_tasks = [load_task_spec(p) for p in sorted(glob.glob(os.path.join(v1, "tier0-*-benign-*.yaml")))]
    train_tasks, held_tasks = split_tasks(all_tasks)
    print(f"train scenarios: {sorted({scenario_of(t.id) for t in train_tasks})}  held-out: {sorted(HELD_OUT)}")

    needs_attacker = any(t.episode_type == "attack" for t in all_tasks)
    if needs_attacker:
        atk_tok = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
        atk_model = AutoModelForCausalLM.from_pretrained(ATTACKER_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")

        def hf_attacker_generate(system, messages):
            chat = [{"role": "system", "content": system}, *messages]
            ids = atk_tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = atk_model.generate(ids, attention_mask=torch.ones_like(ids),
                                         do_sample=True, temperature=0.7, max_new_tokens=256)
            return atk_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

        def attacker_factory(goal):
            return PromptedAttacker(generate=hf_attacker_generate, goal=goal)
    else:
        attacker_factory = None

    for it in range(N_ITERS):
        cap.buffer.clear()
        model.eval()
        examples = rollout(train_tasks, attacker_factory, defender_factory,
                           LocalSandbox, cap, n_rollouts=N_ROLLOUTS)
        assign_advantages(examples)

        from collections import defaultdict
        task_rewards = defaultdict(dict)
        for ex in examples:
            task_rewards[ex.task_id][ex.episode_id] = ex.reward
        for tid, ep_r in sorted(task_rewards.items()):
            print(f"  {tid}: rewards={list(ep_r.values())}")

        model.train()
        metrics = update_step(learner, examples)

        # dashboard: reward decomposition on the train benign track (from this iteration's rollouts)
        # (compute from a fresh no-grad eval pass or reuse rollout results; see note)
        print(f"iter {it:3d}  loss={metrics['loss']:.4f}  ratio={metrics['mean_ratio']:.3f}  "
              f"live={metrics['n_live_examples']}/{metrics['n_examples']}")

        if it % 5 == 0:
            model.save_pretrained(f"/runs/defender-grpo/iter{it}")
            runs.commit()

    model.save_pretrained("/runs/defender-grpo/final")
    runs.commit()


@app.local_entrypoint()
def main():
    train.remote()