# results_logger.py
import pandas as pd
import os

RESULT_PATH = "D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/model_results.csv"

def log_result(result_input, model_name, set_name):

    if isinstance(result_input, pd.DataFrame):
        df_new = result_input.copy()
        df_new["model"] = model_name
        df_new["set"] = set_name
        cols = ["model", "set"] + [c for c in df_new.columns if c not in ["model", "set"]]
        df_new = df_new[cols]
    elif isinstance(result_input, dict):
        rows = []
        for cls, metrics in result_input.items():
            if isinstance(metrics, dict):
                row = {
                    "model": model_name,
                    "set": set_name,
                    "class": cls,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1-score": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                }
            else:   
                row = {
                    "model": model_name,
                    "set": set_name,
                    "class": cls,
                    "precision": metrics,
                    "recall": metrics,
                    "f1-score": metrics,
                    "support": None
                }
            rows.append(row)
        df_new = pd.DataFrame(rows)

    if os.path.exists(RESULT_PATH) and os.path.getsize(RESULT_PATH) > 0:
        df_existing = pd.read_csv(RESULT_PATH)
        df_existing = df_existing[~((df_existing["model"] == model_name) & (df_existing["set"] == set_name))]
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(RESULT_PATH, index=False)

    print(f"✅ Saved results for '{model_name}' ({set_name}) → {RESULT_PATH}")
