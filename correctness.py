import pickle
import argparse
from entailment import get_gpt_entailment, get_deberta_entailment, get_oneshot_gpt_correctness
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(
                    prog='Compute correctness',
                    description='Post clustering entailment to decide response correctness',
                    epilog='')

parser.add_argument('--model', default="openai", type=str,
                    choices=["openai"])     

parser.add_argument('--temp', default=1.0, type=float)     

parser.add_argument('--reasoning', action='store_true')     

parser.add_argument('--oneshot', action='store_true')     

parser.add_argument('--entailment', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

parser.add_argument('--checker', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

args = parser.parse_args()

with open(f'./data/{args.model}_temp={args.temp}_reasoning={args.reasoning}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{args.model}_{args.entailment}_oneshot={args.oneshot}_reas={args.reasoning}_temp={args.temp}_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

collected_correctness = []

if args.checker == "gpt":
    ENTAILER = get_gpt_entailment
else:
    ENTAILER = get_deberta_entailment


def process_sequence(s):
    print(s["id"])
    idx = s["id"]
    question = s["question"]
    answers = s["generated_answers"]
    perplexity = s["generated_perplexity"]
    true_answer = s["true_answer"]
    semantic_set = semantic_set_ids[idx] # dict[id, id]

    # find the largest semantic set
    answer_labels = [semantic_set[i] for i in range(len(answers))]
    labels, label_counts = np.unique(answer_labels, return_counts=True)
    # sort from largest to smallest cluster
    labels = labels[np.argsort(-label_counts)]
    label_counts = label_counts[np.argsort(-label_counts)]

    cluster_correct_strict = False
    cluster_correct_relaxed = False
    cluster_correct_majority = False
    cluster_correct_lowest = False
    cluster_correct_oneshot_all = False
    cluster_correct_oneshot_most = False

    # check lowest perplexity answer entailment
    lowest_perp_answer = answers[np.argmin(perplexity)]
    perplexity_correct = ENTAILER(question, lowest_perp_answer, true_answer)

    if len(labels) == 1:
        # check true answer against all answers
        entailed = [ENTAILER(question, answer, true_answer) for answer in answers]
        cluster_correct_strict = np.all(entailed)
        cluster_correct_relaxed = np.any(entailed)
        cluster_correct_majority = (np.sum(entailed) / len(entailed)) > 0.5
        cluster_correct_lowest = perplexity_correct # true when there is only one semantic group
        cluster_correct_oneshot_all, cluster_correct_oneshot_most = get_oneshot_gpt_correctness(
            question, true_answer, answers
        )
    else:
        # check for a tie in largest label clusters
        if label_counts[0] == label_counts[1]:
            # if there's a tie then answer is wrong/uncertain
            pass
        else:
            answer_subset = [a for a, l in zip(answers, answer_labels) if l == labels[0]]
            perplexity_subset = [a for a, l in zip(perplexity, answer_labels) if l == labels[0]]
            entailed = [ENTAILER(question, answer, true_answer) for answer in answer_subset]
            cluster_correct_strict = np.all(entailed)
            cluster_correct_relaxed = np.any(entailed)
            cluster_correct_majority = (np.sum(entailed) / len(entailed)) > 0.5
            cluster_correct_oneshot_all, cluster_correct_oneshot_most = get_oneshot_gpt_correctness(
                question, true_answer, answers
            )
            
            # check lowest perplexity answer in largest group
            lowest_perp_answer = answer_subset[np.argmin(perplexity_subset)]
            cluster_correct_lowest = ENTAILER(question, lowest_perp_answer, true_answer)

    # check lowest perplexity answer entailment
    lowest_perp_answer = answers[np.argmin(perplexity)]
    perplexity_correct = ENTAILER(question, lowest_perp_answer, true_answer)
    return dict(
        id=idx,
        cluster_correct_strict=cluster_correct_strict,
        cluster_correct_relaxed=cluster_correct_relaxed,
        cluster_correct_majority=cluster_correct_majority,
        cluster_correct_lowest=cluster_correct_lowest,
        cluster_correct_oneshot_all=cluster_correct_oneshot_all,
        cluster_correct_oneshot_most=cluster_correct_oneshot_most,
        perplexity_correct=perplexity_correct,
        question=question,
        answers=answers,
        true_answer=true_answer,
        answer_labels=answer_labels,
        perplexity=perplexity
    )

results = Parallel(n_jobs=10, prefer='threads')(delayed(process_sequence)(s) for s in sequences)
for r in results:
    collected_correctness.append(r)


with open(f'./data/{args.model}_{args.entailment}_oneshot={args.oneshot}_temp={args.temp}_reas={args.reasoning}_checker={args.checker}_correctness.pkl', 'wb') as outfile:
    pickle.dump(collected_correctness, outfile)

