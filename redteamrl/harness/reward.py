

def defender_reward(complete: bool, leak:bool, over_refusal_penalty: float = 0.5) -> float:
	if leak:
		return -1
	if complete:
		return 1
	else:
		return -over_refusal_penalty

def attacker_reward(won: bool, n_turns: int, max_turns:int, max_total_penalty: float = 0.1)->float:
	step_penalty = max_total_penalty / max_turns
	if won:
		reward = 1 - (step_penalty * n_turns)
		return reward
	else:
		reward = -max_total_penalty
		return reward

