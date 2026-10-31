# AI-Powered Student Performance Prediction

Beginner-friendly, offline-capable 10-day AI/ML project built with Python, Pandas, NumPy, Scikit-learn, Matplotlib and Streamlit.

## Important dataset note

This package uses a **synthetic dataset generated for this project**. It is not real student data and should not be described as a public real-world dataset. Synthetic data keeps the demo reproducible and privacy-safe. If your mentor requires a public dataset, replace the CSV with a properly credited dataset and update the data dictionary/report. A possible source is the UCI Student Performance dataset: https://archive.ics.uci.edu/dataset/320/student-performance. Note that its target G3 is strongly correlated with earlier-period grades G1/G2; omit G1/G2 if they would not be available at prediction time.

## Setup (Windows / macOS / Linux)

1. Install Python 3.10 or newer.
2. Extract the ZIP and open a terminal in this folder.
3. Create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Train/evaluate the model: `python train_model.py`
6. Generate EDA figures: `python eda.py`
7. Launch the dashboard: `streamlit run app.py`
8. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Project files

- `data/student_performance_synthetic.csv`: generated, non-identifiable demonstration data
- `train_model.py`: cleanup, train/test split, preprocessing pipeline, baseline comparison, metrics and saved model
- `eda.py`: reproducible EDA figures
- `app.py`: prediction UI and separate rule-based recommendations
- `models/`: saved trained pipeline
- `outputs/`: charts and evaluation metrics
- `report/project_report.pdf`: project report
- `screenshots/`: visual assets for report/demo preparation

## Leakage prevention

The target `final_score` is removed from X. The test split is created before fitting imputation/encoding. The preprocessing and estimator are in one scikit-learn Pipeline, so imputation and category encoding learn from training data only. No final score or derived target label is used as an input.

## Metrics

MAE and RMSE are measured in score points; lower is better. R² compares the model with a mean-based reference and may be negative on difficult test sets. The random split is reproducible with random_state=42.

## Ethical use and limitations

This is an educational prototype trained on synthetic patterns. Metrics do not establish real-world validity. Predictions can be wrong and should never determine grades, admissions, discipline, or access to support. Recommendations are transparent rules, separate from model predictions. Do not enter identifiable student information.

## Submission checklist

- Python source and requirements
- Synthetic dataset and data dictionary
- Training/evaluation pipeline and saved model
- Streamlit application
- EDA charts
- Project report PDF
- Capture at least 3 screenshots from your own running app
- Record an approximately 2-minute demo showing inputs, prediction, recommendations and charts
- Create your GitHub repository and paste its real URL into your submission

