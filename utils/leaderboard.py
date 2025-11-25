"""
CUDAgraveyard utils: Public leaderboard for kills
"""

import json
import os

LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []

def save_leaderboard(leaderboard):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=2)

def update_leaderboard(kill_name, improvement, date, gpu="RTX4090"):
    """
    Add a kill to leaderboard.
    improvement: e.g " +25% TFLOPS -40% watts"
    """
    leaderboard = load_leaderboard()
    entry = {
        "kill": kill_name,
        "improvement": improvement,
        "date": date,
        "gpu": gpu
    }
    leaderboard.append(entry)
    save_leaderboard(leaderboard)
    print(f"Leaderboard updated: {entry}")

if __name__ == "__main__":
    update_leaderboard("GEMM 8192", "+12% TFLOPS", "2025-11-25")
