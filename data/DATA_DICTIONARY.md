# Data dictionary

Dataset: `student_performance_synthetic.csv` (synthetic, generated with NumPy seed 42). No real people or records are represented.

| Column | Meaning | Range / values |
|---|---|---|
| study_hours_per_week | Study hours in a week | 1–20 in supplied sample |
| attendance_percent | Attendance percentage | 35–100 |
| previous_exam_score | Previous exam score | 10–100 |
| assignments_completed | Assignments completed out of 10 | 0–10 |
| sleep_hours_per_night | Average nightly sleep | approximately 4–10 |
| class_participation | Participation scale | 1–5 |
| internet_access | Internet access at home | Yes / No |
| extra_support | Extra learning support | Yes / No |
| final_score | Synthetic outcome target | 0–100; never used as an input |

The CSV intentionally contains a small number of missing feature values and duplicate rows so preprocessing can be demonstrated. Target-missing rows are excluded during training.
