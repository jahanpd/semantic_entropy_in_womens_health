import os
import pickle
import numpy as np
import pandas as pd
from prompt_utils import get_openai_response
import argparse
from typing import TypedDict
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(
                    prog='generate',
                    description='Script 1: Generate M answers to questions from the MRCOG question bank',
                    epilog='')

parser.add_argument('--M', default=10, type=int)     
parser.add_argument('--temp', default=1.0, type=float)     
parser.add_argument('--model', default="openai", type=str,
                    choices=["openai", "llama8b"])     
parser.add_argument('--reasoning', action='store_true')     

args = parser.parse_args()

print(args)

class Output(TypedDict):
    id: int
    category: str
    generated_answers: list[str]
    generated_logprobs: list[list[float]]
    generated_perplexity: list[float]
    true_answer: str
    question: str

SaveData = list[Output]

# number of inference runs (answers) to generate
GENERATIONS = args.M

# import question bank
QUESTIONS = pd.read_csv(os.environ["DATASET_PATH"], index_col=0)
MODEL = args.model
API_RESPONSE = get_openai_response

print(list(QUESTIONS))

def get_answer_text(row):
        return row[f"Option {row['Actual Answer']}"]

result: SaveData = []

if MODEL == "openai":
    def process_row(index, row):
        question = row['Question']
        answer = get_answer_text(row)
        id=index
        question=question
        true_answer=answer
        generated_answers = []
        generated_reasoning = []
        generated_logprobs = []
        generated_perplexity = []
        print("{}/{}: {}".format(index, QUESTIONS.shape[0], question))
        for i in range(GENERATIONS):
            res_struct, res_logprobs, category = API_RESPONSE(
                question,
                reasoning=args.reasoning,
                temperature=args.temp
            )
            print(index, res_struct)
            res_ans = res_struct.short_answer
            if category == "reasoning":
                res_reas = res_struct.reasoning
            else:
                res_reas = None
            generated_answers.append(res_ans)
            generated_reasoning.append(res_reas)
            generated_logprobs.append(res_logprobs)
            perplexity = np.exp(-np.mean(res_logprobs))
            generated_perplexity.append(perplexity)

        output: Output = dict(
            id=id,
            question=question,
            category=category,
            true_answer=true_answer,
            generated_answers=generated_answers,
            generated_reasoning=generated_reasoning,
            generated_logprobs=generated_logprobs,
            generated_perplexity=generated_perplexity
        )
        return output

    results = Parallel(n_jobs=10, prefer='threads')(delayed(process_row)(index, row) for index, row in QUESTIONS.iterrows())

    for r in results:
        if r is not None:
            result.append(r)

elif MODEL == "llama8b":
    pass


with open(f'./data/{MODEL}_temp={args.temp}_reasoning={args.reasoning}_generations.pkl', 'wb') as outfile:
    pickle.dump(result, outfile)
