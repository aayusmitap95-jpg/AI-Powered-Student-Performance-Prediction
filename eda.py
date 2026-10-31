"""Create explanatory EDA figures from the supplied synthetic dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "student_performance_synthetic.csv").drop_duplicates()
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
plt.figure(figsize=(7,4.5)); df["final_score"].hist(bins=20)
plt.title("Distribution of Final Scores"); plt.xlabel("Final score (0–100)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(out/"score_distribution.png",dpi=150); plt.close()
for col, title, xlabel in [
    ("study_hours_per_week","Study Hours vs Final Score","Study hours per week"),
    ("attendance_percent","Attendance vs Final Score","Attendance (%)")]:
    plt.figure(figsize=(7,4.5)); plt.scatter(df[col],df["final_score"],alpha=.45)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Final score")
    plt.tight_layout(); plt.savefig(out/(col+"_vs_score.png"),dpi=150); plt.close()
corr = df.select_dtypes(include="number").corr()
plt.figure(figsize=(8,6)); plt.imshow(corr, aspect="auto"); plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns,rotation=75,ha="right",fontsize=8)
plt.yticks(range(len(corr.columns)),corr.columns,fontsize=8)
plt.title("Numeric Feature Correlation Matrix"); plt.tight_layout()
plt.savefig(out/"correlation_matrix.png",dpi=150); plt.close()
print("EDA figures saved in outputs/")
