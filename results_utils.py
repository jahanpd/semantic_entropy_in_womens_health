from pydantic import BaseModel
import os
from enum import Enum
import pickle
import pandas as pd
import numpy as np
from typing import Optional
from roc import *
from sklearn.metrics import roc_auc_score
import itertools
import random

def confidence_interval(values):
    mean = np.mean(values)
    sem = np.std(values) / np.sqrt(len(values))  
    # Calculate 95% CI
    z = 1.96  # 95% confidence level
    ci_lower = mean - z * sem
    ci_upper = mean + z * sem
    return ci_lower, ci_upper


class Entail(Enum):
    GPT = 0
    DEBERTA = 1
    ONESHOT = 2

class Result(BaseModel):
    temp: float
    reasoning: bool
    entailment: Entail
    checker: Entail
    confidence: Optional[list[dict]] = None
    correctness: Optional[list[dict]] = None
    category: Optional[list[str]] = None
    length: Optional[list[float]] = None

class Results:
    def __init__(self, results: list[Result], dataset_path: str):
        assert len(results) > 0

        questions = pd.read_csv(dataset_path, index_col=0)
        self.questions = questions
        self.parts = {
            "part1": self.check_part1,
            "part2" : self.check_part2,
            "knowledge": self.check_knowledge,
            "reasoning": self.check_reasoning,
            "short": self.check_short,
            "long": self.check_long,
            "full" : lambda x, *args: True,
            }

        for r in results:
            path = f'./data/openai_{self.entail_str(r.entailment)}_oneshot={self.oneshot(r.entailment)}_temp={r.temp}_reas={r.reasoning}_agg=original_confidence.pkl'
            print(path)
            with open(path, 'rb') as infile:
                res = pickle.load(infile)
                r.confidence = [item for item in res if self.filter(item["ids"])]

            path = f'./data/openai_{self.entail_str(r.entailment)}_oneshot={self.oneshot(r.entailment)}_temp={r.temp}_reas={r.reasoning}_checker={self.entail_str(r.checker)}_correctness.pkl'
            print(path)
            with open(path, 'rb') as infile:
                res = pickle.load(infile)
                r.correctness = [item for item in res if self.filter(item["id"])]

            path = f'./data/openai_temp={r.temp}_reasoning={r.reasoning}_generations.pkl'
            print(path)
            with open(path, 'rb') as infile:
                res = pickle.load(infile)
                r.category = [item["category"] for item in res if self.filter(item["id"])]
                r.length = [np.mean([len(x) for x in item["generated_answers"]]) for item in res if self.filter(item["id"])]
                print(min(r.length), np.mean(r.length), max(r.length))

        self.results: list[Result] = results
        print([np.sum([cat == "knowledge" for cat in r.category]) for r in results])
        print([np.sum([cat == "reasoning" for cat in r.category]) for r in results])

    def entail_str(self, entail: Entail):
        return "gpt" if entail == Entail.GPT or entail == Entail.ONESHOT else "deberta"

    def oneshot(self, entail: Entail):
        return entail == Entail.ONESHOT
    
    def filter(self, id, *args):
        return True
    
    def check_part1(self, id, *args):
        return self.questions.loc[id].Part == "One"

    def check_part2(self, id, *args):
        return self.questions.loc[id].Part == "Two"
    
    def check_knowledge(self, id, *args):
        return args[0] == "knowledge"

    def check_reasoning(self, id, *args):
        return args[0] == "reasoning"
    
    def check_short(self, id, *args):
        return args[1] < 15

    def check_long(self, id, *args):
        return args[1] > 60

    def filter_results(self, 
            temp=[0.2, 1.0, 1.1], 
            reasoning=[True, False],
            entailment=[Entail.GPT, Entail.DEBERTA]):
        filter = [r for r in self.results if r.temp in temp and r.reasoning in reasoning and r.entailment in entailment]
        names = [f'Temp={r.temp}|Reasoning={r.reasoning}|entailed with {self.entail_str(r.entailment)}' for r in filter]
        return filter, names

    def get_results_df(self, steps=1000, seed=42):
        random.seed(seed)
        df = {
            "temp": [],
            "reasoning": [],
            "entailment": [],
            "checker": [],
            "metric": [],
            "correctness": [],
            "part": [],
            "acc": [],
            "acc_ci_l": [],
            "acc_ci_u": [],
            "auc": [],
            "auc_ci_l": [],
            "auc_ci_u": [],
        }
        metrics = ['entropy', 'dentropy']
        part = ['full', 'part1', 'part2', 'knowledge', 'reasoning', 'short', 'long']
        correct_definition = [
            'cluster_correct_strict', 'cluster_correct_relaxed', 
            'cluster_correct_majority', 'cluster_correct_lowest',
            'cluster_correct_oneshot_all', 'cluster_correct_oneshot_most'
            ]
        combinations = list(itertools.product(metrics, part, correct_definition))
        perp_combo = list(itertools.product(['perplexity'], part, ['perplexity_correct']))

        for r in self.results:
            for mname, pname, cname in combinations+perp_combo:

                p = ([item[mname] for item, cat, length in zip(r.confidence, r.category, r.length) if self.parts[pname](item["ids"], cat, length)],
                    [item[cname] for item, cat, length in zip(r.correctness, r.category, r.length) if self.parts[pname](item["id"], cat, length)])

                try:
                    df["temp"].append(r.temp)
                    df["reasoning"].append(r.reasoning)
                    df["entailment"].append(r.entailment)
                    df["checker"].append(r.checker)
                    df["metric"].append(mname)
                    df["correctness"].append(cname)
                    df["part"].append(pname)

                    acc, ci_lower, ci_upper = self.accuracy(p[1])
                    df["acc"].append(np.mean(acc))
                    df["acc_ci_l"].append(ci_lower)
                    df["acc_ci_u"].append(ci_upper)

                    auc, ci_lower, ci_upper = self.auroc(p[0], p[1])
                    df["auc"].append(auc)
                    df["auc_ci_l"].append(ci_lower)
                    df["auc_ci_u"].append(ci_upper)

                except Exception as e:
                    print(np.sum(np.isnan(p[0])))
                    print(mname, pname, cname)

        return pd.DataFrame(df).drop_duplicates()


    
    def accuracy(self, correct):
        arr = np.array(correct).astype(np.float32)
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        p = arr.sum() / arr.shape[0]
        n = len(arr)
        z = 1.96
        d = 1 + z**2/n
        cap = p + z*z / (2*n)
        asd = np.sqrt((p*(1 - p) + z*z / (4*n)) / n)

        lower = (cap - z*asd) / d
        upper  = (cap + z*asd) / d
        return p, lower, upper

    def auroc(self, score, correct):
        AUC = roc_auc_score(
            np.array(correct).astype(np.float32),
            -1*np.array(score).astype(np.float32)
        )
        # Hanley and McNeil, The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology (1982) 43 (1) pp. 29-36.
        N1 = np.sum([c for c in correct])
        N2 = np.sum([not c for c in correct])
        Q1 = AUC / (2 - AUC)
        Q2 = 2*AUC**2 / (1 + AUC)
        SE_AUC = np.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
        lower = AUC - 1.96*SE_AUC
        upper = AUC + 1.96*SE_AUC
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return AUC, lower, upper

    def plot_aurocs_sem_ent_full_gpt(self, title=""):
        """Plot AUROC curves for Semantic Uncertainty subset by LLM entailment"""
        res, names = self.filter_results(
            entailment=[Entail.GPT]
        )

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        su_rocs_from_results(
            res, 
            ax1,
            names,
            title
            )
            
    def plot_aurocs_temp(self, title=""):
        """Plot AUROC curves for Semantic Uncertainty stratified by temperature"""
        res = [r for r in self.results if 
                  r.temp in [0.2, 1.0] and
                  r.reasoning == False and
                  r.entailment == Entail.GPT and
                  r.checker == Entail.GPT]
        names = [f'Temp={r.temp}' for r in res]

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        su_rocs_from_results(
            res, 
            ax1,
            names,
            title
            )

    def plot_aurocs_category(self, parts=('reasoning', 'knowledge'), title=""):
        """Plot AUROC curves for Semantic Uncertainty by category"""

        result = [r for r in self.results if 
                  r.temp in [1.0] and
                  r.reasoning == False and
                  r.entailment == Entail.GPT and
                  r.checker == Entail.GPT][0]
        res = [Result(
            temp=result.temp, 
            reasoning=result.reasoning,
            entailment=result.entailment,
            checker=result.checker,
            confidence=[x for x,cat in zip(result.confidence, result.category) if cat == part],
            correctness=[x for x,cor in zip(result.correctness, result.category) if cor == part]
            ) for part in parts]
        names = [part for part in parts]

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        su_rocs_from_results(
            res, 
            ax1,
            names,
            title
            )

    def plot_aurocs_length(self, parts=('short', 'long'), title=""):
        """Plot AUROC curves for Semantic Uncertainty by category"""

        result = [r for r in self.results if 
                  r.temp in [1.0] and
                  r.reasoning == False and
                  r.entailment == Entail.GPT and
                  r.checker == Entail.GPT][0]
        res = [Result(
            temp=result.temp, 
            reasoning=result.reasoning,
            entailment=result.entailment,
            checker=result.checker,
            confidence=[x for x,length in zip(result.confidence, result.length) if self.parts[part](None, None, length)],
            correctness=[x for x,length in zip(result.correctness, result.length) if self.parts[part](None, None, length)],
            ) for part in parts]
        names = [part for part in parts]

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        su_rocs_from_results(
            res, 
            ax1,
            names,
            title
            )

    def plot_aurocs_metrics_standard(self, title="This is a title"):
        """Plot AUROC curves for all metrics in the base case of temp=1.0 and no reasoning"""
        res, names = self.filter_results(
            entailment=[Entail.GPT],
            temp=[1.0],
            reasoning=[False]
        )

        _, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        rocs_from_results(
            results_array=res,
            axes=[ax1 for _ in range(len(res))],
            titles=names
        )
        ax1.set_title(title)
