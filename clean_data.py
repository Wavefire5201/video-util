import json

with open("results2.json", "r") as f:
    results = json.load(f)

previous_weight = float("inf")
weights = []

for result in results:
    weights.append(float(result["weight"]))
    if result["weight"] > previous_weight:
        print(result["date"])
    previous_weight = result["weight"]

print(sum(weights) / len(weights))
