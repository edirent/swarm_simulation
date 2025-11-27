import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_log(path):
    with open(path, "r") as f:
        return json.load(f)


def main(log_path="logs/sim.json"):
    data = load_log(log_path)
    ts = [entry["t"] for entry in data]
    coverage = []
    for entry in data:
        if not entry["agents"]:
            coverage.append(0.0)
            continue
        positions = np.array([a["pos"] for a in entry["agents"].values()])
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        coverage.append(float(np.prod(maxs - mins)))

    plt.plot(ts, coverage)
    plt.xlabel("time")
    plt.ylabel("coverage (bbox area)")
    plt.title("Swarm coverage over time")
    plt.show()


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "logs/sim.json"
    main(log)
