import json

import pandas as pd
import ast

from config import RUN_LOGS, DATA_DIR
from workflow.run_workflow import run_workflow_func

FLAIR_DICT = {
    "Not the A-hole": "NTA",
    "Asshole": "YTA"
}


def run_eval(run_name):
    data_path = DATA_DIR / "hundred_test.json"
    with open(data_path, "r") as data_file:
        data = json.load(data_file)

    cols = ["post_id", "title", "post_link", "flair", "verdict_short", "verdict_correct", "selftext", "opposite_text", "debate", "debate_CR", "verdict", "verdict_pct",
            "verdict_G", "advice", "advice_AR", "advice_G"]
    rows = []
    df = pd.DataFrame(rows, cols)

    for i, record in enumerate(data):
        if i > 10:
            break
        flair = FLAIR_DICT[record["flair"]]
        rec_result = [record["post_id"], record["title"], record["url"], flair]
        verdict_short, opposite_text, debate, debate_CR, verdict, verdict_pct, verdict_G, advice, advice_AR, advice_G= run_workflow_func(run_name, record)
        rec_result += [verdict_short, verdict_short == flair, record["selftext"], opposite_text, debate, debate_CR, verdict, verdict_pct, verdict_G, advice, advice_AR, advice_G]
        rows.append(rec_result)

        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(RUN_LOGS / f"{run_name}/result.csv", index=False)
    return df

def eval_df(df):
    acc = df["verdict_correct"].mean()
    tp = df[(df["flair"] == "YTA") & (df["verdict_short"] == "YTA")].count()[0]
    fp = df[(df["flair"] == "NTA") & (df["verdict_short"] == "YTA")].count()[0]
    tn = df[(df["flair"] == "NTA") & (df["verdict_short"] == "NTA")].count()[0]
    fn = df[(df["flair"] == "YTA") & (df["verdict_short"] == "NTA")].count()[0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    try:
        debate_CR_mean = df["debate_CR"].apply(lambda x: x["score"]).mean()
        verdict_G_mean = df["verdict_G"].apply(lambda x: x["score"]).mean()
        advice_AR_mean = df["advice_AR"].apply(lambda x: x["score"]).mean()
        advice_G_mean = df["advice_G"].apply(lambda x: x["score"]).mean()
    except Exception as e:
        debate_CR_mean = df["debate_CR"].apply(lambda x: ast.literal_eval(x)["score"]).mean()
        verdict_G_mean = df["verdict_G"].apply(lambda x: ast.literal_eval(x)["score"]).mean()
        advice_AR_mean = df["advice_AR"].apply(lambda x: ast.literal_eval(x)["score"]).mean()
        advice_G_mean = df["advice_G"].apply(lambda x: ast.literal_eval(x)["score"]).mean()

    result_data = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "debate_CR": debate_CR_mean,
        "verdict_G": verdict_G_mean,
        "advice_AR": advice_AR_mean,
        "advice_G": advice_G_mean,
    }
    with open(RUN_LOGS / f"{run_name}/result.json", "w") as f:
        json.dump(result_data, f, indent=4)


if __name__ == '__main__':
    run_name = "run_nta2"
    # df = run_eval(run_name)
    df = pd.read_csv(RUN_LOGS / f"{run_name}/result.csv")
    eval_df(df)
