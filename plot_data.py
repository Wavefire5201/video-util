import numpy as np
import json
import datetime
import matplotlib.pyplot as plt

with open("results2.json", "r") as f:
    results = json.load(f)

dates = []
weights = []

for result in results:
    date = datetime.datetime.fromtimestamp(float(result["date"]))
    dates.append(date)
    weights.append(result["weight"])

plt.figure(figsize=(10, 5))
plt.plot(dates, weights, marker="o")
plt.title("Weight Over Time")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
