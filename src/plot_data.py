import numpy as np
import json
import datetime
import matplotlib.pyplot as plt

with open("3-3_to_3-4_Model_2015.json", "r") as f:
    results = json.load(f)

dates = []
weights = []

for result in results:
    date = datetime.datetime.fromtimestamp(float(result["date"]))
    dates.append(date)
    weights.append(result["weight"] * -1)

plt.figure(figsize=(10, 5))
plt.plot(dates, weights, marker="o")
plt.title("Weight Over Time")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
