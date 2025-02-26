import pickle
from entailment import SemanticSet, get_deberta_entailment, get_gpt_entailment, get_oneshot_gpt_entailment
import argparse
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

parser = argparse.ArgumentParser(
                    prog='Semantic Similarity',
                    description='Script 2: Measure semantic similarity and cluster into sets for generated and true answers',
                    epilog='')

parser.add_argument('--model', default="openai", type=str,
                    choices=["openai"])     

parser.add_argument('--temp', default=1.0, type=float)     

parser.add_argument('--reasoning', action='store_true')     

parser.add_argument('--oneshot', action='store_true')     

parser.add_argument('--entailment', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

args = parser.parse_args()

# semantic set is dict[int, int]
SetSemanticSets = dict[int, SemanticSet]

MODEL = args.model
ENTAILMENT = args.entailment

with open(f'./data/{MODEL}_temp={args.temp}_reasoning={args.reasoning}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

semantic_sets: SetSemanticSets = {}

if args.entailment == "deberta":
    for s in sequences:
        print("id ", s['id'], "deberta")
        question = s["question"]
        answers = s["generated_answers"]
        # base semantic set 
        semantic_set_ids: SemanticSet = {}
        for idx, answer in enumerate(answers):
            # initialize with a bad answer
            semantic_set_ids[idx] = -1

        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(answers):
            # this inner loop compared each gen ans with other answers

            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                semantic_set_ids[i] = next_id
                # If string1 has not been assigned an id, assign it next_id.
                for j in range(i + 1, len(answers)):
                    entailed = get_deberta_entailment(
                            question,
                            string1,
                            answers[j],
                            strict=True
                            )
                    if entailed:
                        semantic_set_ids[j] = semantic_set_ids[i]
                next_id += 1

        semantic_sets[s['id']] = semantic_set_ids


if args.entailment == "gpt" and not args.oneshot:
    def process_sequence(s):
        print("id ", s['id'], "gpt")
        question = s["question"]
        answers = s["generated_answers"]
        # base semantic set 
        semantic_set_ids: SemanticSet = {}
        for idx, answer in enumerate(answers):
            # initialize with a bad answer
            semantic_set_ids[idx] = -1

        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(answers):
            # this inner loop compared each gen ans with other answers

            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                semantic_set_ids[i] = next_id
                # If string1 has not been assigned an id, assign it next_id.
                for j in range(i + 1, len(answers)):
                    entailed = get_gpt_entailment(
                            question,
                            string1,
                            answers[j],
                            strict=True
                            )
                    if entailed:
                        semantic_set_ids[j] = semantic_set_ids[i]
                next_id += 1

        return (s['id'], semantic_set_ids)

    results = Parallel(n_jobs=10, prefer='threads')(delayed(process_sequence)(s) for s in sequences)
    for idx, ssid in results:
        semantic_sets[idx] = ssid
    
if args.entailment == "gpt" and args.oneshot:
    def process_sequence(s):
        print("id ", s['id'], "gpt")
        question = s["question"]
        answers = s["generated_answers"]
        # base semantic set 

        semantic_set_ids = get_oneshot_gpt_entailment(question, answers)
        return (s['id'], semantic_set_ids)

    results = Parallel(n_jobs=10, prefer='threads')(delayed(process_sequence)(s) for s in sequences)
    for idx, ssid in results:
        semantic_sets[idx] = ssid
 
with open(f'./data/{MODEL}_{ENTAILMENT}_oneshot={args.oneshot}_reas={args.reasoning}_temp={args.temp}_semantic_similarity.pkl', 'wb') as outfile:
    pickle.dump(semantic_sets, outfile)
