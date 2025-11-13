import json

import pandas as pd

from config import RUN_LOGS
from workflow.workflow import run_workflow


FLAIR_DICT = {
    "Not the A-hole": "NTA",
    "Asshole": "YTA"
}


def run_eval(run_name):
    data_path = ""
    with open(data_path, "r") as data_file:
        data = json.load(data_file)

    cols = ["post_id", "title", "post_link", "flair", "verdict_short", "verdict_correct", "selftext", "opposite_text", "debate", "debate_CR", "verdict", "verdict_pct",
            "verdict_G", "advice", "advice_AR", "advice_G"]
    rows = []
    for record in data:
        flair = FLAIR_DICT[record["flair"]]
        rec_result = [record["post_id"], record["title"], record["url"], flair]
        verdict_short, opposite_text, debate, debate_CR, verdict, verdict_pct, verdict_G, advice, advice_AR, advice_G= run_workflow(run_name, record)
        rec_result += [verdict_short, verdict_short == flair, record["selftext"], opposite_text, debate, debate_CR, verdict, verdict_pct, verdict_G, advice, advice_AR, advice_G]
        rows.append(rec_result)

    df = pd.DataFrame(rows, cols)
    acc = df["verdict_correct"].mean()
    tp = df[(df["flair"] == "YTA") & (df["verdict"] == "YTA")].sum()
    fp = df[(df["flair"] == "NTA") & (df["verdict"] == "YTA")].sum()
    tn = df[(df["flair"] == "NTA") & (df["verdict"] == "NTA")].sum()
    fn = df[(df["flair"] == "NTA") & (df["verdict"] == "YTA")].sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("Results:", acc, precision, recall)

    result_data = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
    }
    json.dump(result_data, RUN_LOGS / f"{run_name}/result.json")


if __name__ == '__main__':
    run_name = "run1"
    run_eval(run_name)
