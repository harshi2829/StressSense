import csv

rows = 70000
filename = "test_4col_70000rows.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ch1", "ch2", "ch3", "ch4"])  # header
    for i in range(rows):
        writer.writerow([i, i + 1, i + 2, i + 3])

print("Saved", filename)
