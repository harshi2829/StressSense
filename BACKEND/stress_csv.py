import csv

rows = 70000
filename = "test_stress_like_4col_70000rows.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ch1", "ch2", "ch3", "ch4"])
    for i in range(rows):
        # Simulate more irregular, higher-amplitude signals
        writer.writerow([
            i % 50 + 50,
            (i * 3) % 70 + 40,
            (i * 7) % 90 + 30,
            (i * 11) % 110 + 20,
        ])

print("Saved", filename)
