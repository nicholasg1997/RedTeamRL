import time

from redteamrl.train.cpu_monitor import label_process, start_cpu_monitor


def test_labels_name_the_processes_that_can_bottleneck_a_rollout():
    assert label_process(["python", "train.py"], is_self=True) == "trainer+rollout harness"
    assert label_process(["/usr/bin/vllm", "serve", "Qwen/Qwen3-8B", "--port", "8001"],
                         is_self=False) == "vllm api :8001"
    assert label_process(["VLLM::EngineCore"], is_self=False) == "vllm engine"
    assert label_process(["/bin/bash", "-c", "ls"], is_self=False) == "sandbox shells"


def test_monitor_logs_and_stops(capsys):
    stop = start_cpu_monitor(interval_s=0.05)
    time.sleep(0.2)
    stop.set()
    out = capsys.readouterr().out
    assert "[cpu] trainer+rollout harness=" in out and "cores>90%" in out
