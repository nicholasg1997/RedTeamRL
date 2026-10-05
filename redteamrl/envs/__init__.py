"""Environments: one shared filesystem yields many matched attack/benign tasks.

An `Environment` is an XBOW-style unit — a realistic mini-app plus a set of attack goals
and benign requests over it. `build_tasks` injects the canary at build time into a chosen
location and expands the templates into concrete `TaskSpec`s that share one filesystem.
This is the confound-balanced matched-pair design at the environment level: attack and
benign traverse identical files, differing only in intent.
"""
from redteamrl.envs.schema import Environment, AttackTemplate, BenignTemplate, build_tasks

__all__ = ["Environment", "AttackTemplate", "BenignTemplate", "build_tasks"]
