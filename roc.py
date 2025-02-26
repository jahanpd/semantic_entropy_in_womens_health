import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from tabulate import tabulate
from pydantic import BaseModel
from typing import Optional
from enum import Enum

class Entail(Enum):
    GPT = 0
    DEBERTA = 1

class Result(BaseModel):
    temp: float
    reasoning: bool
    entailment: Entail
    confidence: Optional[dict] = None
    correctness: Optional[dict] = None

def rocs_from_results(results_array, axes, titles):
    for results, ax, title in zip(results_array, axes, titles):

        sement = RocCurveDisplay.from_predictions(
            np.array([r["cluster_correct_lowest"] for r in results.correctness]).astype(np.float32),
            -1*np.array([r["entropy"] for r in results.confidence]),
            ax=ax,
            color="red",
            name="Semantic Uncertainty"
        )
        discent = RocCurveDisplay.from_predictions(
            np.array([r["cluster_correct_lowest"] for r in results.correctness]).astype(np.float32),
            -1*np.array([r["dentropy"] for r in results.confidence]),
            ax=ax,
            color="orange",
            name="Discrete Semantic Uncertainty"
        )
        perp = RocCurveDisplay.from_predictions(
            np.array([r["perplexity_correct"] for r in results.correctness]).astype(np.float32),
            -1*np.array([r["perplexity"] for r in results.confidence]),
            ax=ax,
            color="green",
            name="Perplexity"
        )

        ax.set_title(title)

def su_rocs_from_results(results_array: list[Result], ax, titles, main_title=""):
    for results, title in zip(results_array, titles):
        
        def get_string(string):
            if string == "entropy":
                return "Semantic Entropy"
            if string == "dentropy":
                return "Discrete Semantic Entropy"
            if string == "perplexity":
                return "Perplexity"
            
        for metric in ["entropy", "dentropy", "perplexity"]:
            _ = RocCurveDisplay.from_predictions(
                np.array([r["cluster_correct_lowest"] for r in results.correctness]).astype(np.float32),
                -1*np.array([r[metric] for r in results.confidence]),
                ax=ax,
                name= get_string(metric) + f" ({title})"
            )
    ax.set_title(main_title)


def table_from_results(results_array, headers):
    table = [
        ["SE"],
        ["SDE" ],
        ["OSE"],
        ["Perp"]
    ]
    for results in results_array:
        ent_correct = np.array(results["entropy_correct"]).astype(np.float32)
        table[0].append(ent_correct.sum() / ent_correct.shape[0])

        dent_correct = np.array(results["dentropy_correct"]).astype(np.float32)
        table[1].append(dent_correct.sum() / dent_correct.shape[0])

        og_ent_correct = np.array(results["og_entropy_correct"]).astype(np.float32)
        table[2].append(og_ent_correct.sum() / og_ent_correct.shape[0])

        perp_correct = np.array(results["perplexity_correct"]).astype(np.float32)
        table[3].append(perp_correct.sum() / perp_correct.shape[0])
 

    headers = ["Metric"] + headers
    print(tabulate(table, headers=headers))