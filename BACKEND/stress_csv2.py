import csv
import math

rows = 70000
filename = "test_stress2_4col_70000rows.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ch1", "ch2", "ch3", "ch4"])
    for i in range(rows):
        t = i / 100.0
        writer.writerow([
            80 + 30 * math.sin(2 * math.pi * 0.7 * t) + 10 * math.sin(2 * math.pi * 5 * t),
            60 + 25 * math.sin(2 * math.pi * 0.9 * t),
            50 + 20 * math.sin(2 * math.pi * 1.1 * t) + 5 * math.sin(2 * math.pi * 8 * t),
            40 + 35 * math.sin(2 * math.pi * 0.5 * t),
        ])

print("Saved", filename)
