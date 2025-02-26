import json
import pickle
from pydantic import BaseModel
import random
from AesEverywhere import aes256
import os
import pandas as pd

# generator to yield clinician
def get_clinician():
    clinicians = [111, 112, 113]
    index = 0
    while True:
        for c in clinicians:
            yield c

clinician = get_clinician()

class Item(BaseModel):
    id: int
    question: str
    generated_answers: list[str]
    clusters: list[int]
    sement: float
    dsement: float
    perplexity: list[float]
    true_answer: str
    clinician: int
    correct_cluster: bool
    correct_perp: bool

path = "data/openai_temp=1.0_reasoning=False_generations.pkl"
path2 = "data/openai_gpt_oneshot=False_reas=False_temp=1.0_semantic_similarity.pkl"
path3 = "data/openai_gpt_oneshot=False_temp=1.0_reas=False_checker=gpt_correctness.pkl"
path4 = "data/openai_gpt_oneshot=False_temp=1.0_reas=False_agg=original_confidence.pkl"
data = pickle.load(open(path, "rb"))
clusters = pickle.load(open(path2, "rb"))
correctness = pickle.load(open(path3, "rb"))
confidence = pickle.load(open(path4, "rb"))

questions = pd.read_csv("~/Jahan_Subset_v2.csv")

output: list[Item] = []
filter = {}
for i in range(1, 11):
    filter[i] = []

for d in data:
    try:
        c = clusters[d["id"]]
        c = [value for value in c.values()]
        marking = [res for res in correctness if res["id"] == d["id"]]
        conf = [res for res in confidence if res["ids"] == d["id"]]
        assert len(marking) == 1
        numc = len(set(c))
        item = Item(
            id = d["id"],
            question = d["question"],
            generated_answers=d["generated_answers"],
            clusters=c,
            sement=conf[0]["entropy"],
            dsement=conf[0]["dentropy"],
            perplexity=[perp if perp < 10000 else 10000 for perp in d["generated_perplexity"]],
            true_answer=d["true_answer"],
            clinician=next(clinician),
            correct_cluster=marking[0]["cluster_correct_lowest"],
            correct_perp=marking[0]["perplexity_correct"]
        )
        filter[numc].append(dict(item))
    except Exception as e:
        print("exception", e)

random.seed(42)

def get_subset(items):
    if len(items) > 20:
        subsample = random.sample(subset, 20)
        return subsample
    else:
        return items 

for i in range(1, 11):
    subset = filter[i]
    count_one = 0
    count_two = 0
    part1 = [item for item in subset if questions.loc[item["id"], :].Part == 'One' and questions.loc[item["id"], :].isnull().Table]
    part2 = [item for item in subset if questions.loc[item["id"], :].Part == 'Two' and questions.loc[item["id"], :].isnull().Table]

    if len(part1) > len(part2):
        part1 = random.sample(part1, len(part2))
    if len(part2) > len(part1):
        part2 = random.sample(part2, len(part1))
    subpart1 = get_subset(part1)
    subpart2 = get_subset(part2)
    output = output + subpart1 + subpart2
    print(i, len(filter[i]), len(subpart1), len(subpart2))

random.shuffle(output)

# limit clinian questions
counts = {111:0, 112:0, 113:0}
new_output = []

correct = {}
for qn in output:
    count = counts[qn["clinician"]]
    if count <= 35:
        new_output.append(qn)
        correct[qn["id"]] = {
            "correct_perp":qn["correct_perp"],
            "correct_cluster":qn["correct_cluster"],
            "sement":qn["sement"],
            "dsement":qn["dsement"]
            }
        counts[qn["clinician"]] = count + 1


print(correct)
print(len(new_output))

jsonstring = json.dumps(new_output, indent=4)
print(jsonstring[:50])

encrypted = aes256.encrypt(jsonstring, os.environ["SECRET_KEY"])

print(encrypted[:50])

with open('data/encrypted.json', 'w') as f:
    f.write(encrypted.decode('utf-8'))

with open(f'correct.pickle', 'wb') as outfile:
    pickle.dump(correct, outfile)