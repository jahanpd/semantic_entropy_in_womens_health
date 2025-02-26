import pickle
import torch
import numpy as np
import argparse

"""
We use output from previous two scripts to compute confidence values.
"""

parser = argparse.ArgumentParser(
                    prog='Compute confidence',
                    description='Script 3: Measure semantic uncertainty (entropy), discrete SE, and perplexity',
                    epilog='')

parser.add_argument('--model', default="openai", type=str,
                    choices=["openai"])     

parser.add_argument('--temp', default=1.0, type=float)     

parser.add_argument('--reasoning', action='store_true')     

parser.add_argument('--oneshot', action='store_true')     

parser.add_argument('--entailment', default="gpt", type=str,
                    choices=["gpt", "deberta"])     

parser.add_argument('--agg', default="original", type=str,
                    choices=["sum_normalized", "original"])     

args = parser.parse_args()

MODEL = args.model
ENTAILMENT = args.entailment

def logsumexp_by_id(
        semantic_ids: dict[int, int], 
        log_likelihoods: list[float], 
        agg='sum_normalized'
        ):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    # https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L208
    # note that semantic_set_id 0 is the true answer
    assert len(log_likelihoods) == len(semantic_ids), f"{len(log_likelihoods)} vs {len(semantic_ids)} {semantic_ids}" # semantic sets include true answer
    log_likelihood_per_semantic_id = []
    # need to filter out the true answer as no logliks for it, has poisiton/id 0
    unique_ids = sorted(list(set([val for key, val in semantic_ids.items()])))
    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in semantic_ids.items() if x == uid]
        # Gather mean log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices] # list[float]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = np.array(id_log_likelihoods) - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)) + 1e-8)
        elif agg == 'original':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods)) + 1e-8)
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id, unique_ids


def categorical_empirical_loglik(
        semantic_ids: dict[int, int], 
        log_likelihoods: list[float], 
        ):
    """ Calculate counts for each set.

    Return logprob of each set"""
    unique, counts = np.unique([val for key, val in semantic_ids.items()], return_counts=True)
    return [np.log(c/np.sum(counts)) for c in counts], unique
    

def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def compute_semantic_uncertainty(
        log_likelihoods: list[list[float]], 
        semantic_ids: dict[int, int],
        agg='sum_normalized'
        ) -> tuple[float, bool, int]:

    # Length normalization of generation probabilities.
    # log_liks_agg = [np.mean(log_lik) for log_lik in log_likelihoods]
    log_liks_agg = [np.log(np.sum(np.exp(li - np.log(len(li) + 1e-8)))) for li in log_likelihoods]

    # returns list[float] of likelihoods where index corresponds
    # to unique semantic id, and list[int] of semantic ids
    # index 0 is the true answer semantic set if index 0 of unique_ids is 0
    log_likelihood_per_semantic_id, unique_ids_cont = logsumexp_by_id(
            semantic_ids, 
            log_liks_agg, 
            agg=agg)
    log_likelihood_per_semantic_id_discrete, unique_ids_disc = categorical_empirical_loglik(
            semantic_ids, 
            log_likelihoods)
    pe_continuous = predictive_entropy_rao(log_likelihood_per_semantic_id)
    pe_discrete = predictive_entropy_rao(log_likelihood_per_semantic_id_discrete)

    return (pe_continuous, pe_discrete, len(unique_ids_cont), len(unique_ids_disc))

def compute_perplexity(perplexity) -> float:
    perplexity = torch.tensor(perplexity).clamp(0.0, 1000000.0)

    return float(perplexity.min())

with open(f'./data/{MODEL}_temp={args.temp}_reasoning={args.reasoning}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'./data/{MODEL}_{ENTAILMENT}_oneshot={args.oneshot}_reas={args.reasoning}_temp={args.temp}_semantic_similarity.pkl', 'rb') as infile:
    semantic_set_ids = pickle.load(infile)

final_results = []

for sequence in sequences:
    log_likelihoods = sequence['generated_logprobs']
    semantic_set = semantic_set_ids[sequence['id']] # dict[id, id]
    print(sequence['id'])
    # for each question, we calculate
    # 1. semantic uncertainty (entropy)
    # 2. if the highest count set contains the correct answer
    # 3. the number of semantic sets in the generated answers
    # 4. the lowest perplexity 
    # 5. if the lowest perplexity answer is in the same set as the correct answer - aka correct

    entropy, entropy_discrete, sets, sets_disc = compute_semantic_uncertainty(
            log_likelihoods, 
            semantic_set,
            agg=args.agg
            )
    perplexity = compute_perplexity(sequence['generated_perplexity']) 

    final_results.append(dict(
        ids = sequence['id'],
        entropy = entropy,
        dentropy = entropy_discrete,
        sets = sets,
        dsets = sets,
        perplexity = perplexity,
        ))
    print(f"e:{entropy:.2f} ed:{entropy_discrete:.2f} s:{sets} sd:{sets_disc}")
    
with open(f'./data/{MODEL}_{ENTAILMENT}_oneshot={args.oneshot}_temp={args.temp}_reas={args.reasoning}_agg={args.agg}_confidence.pkl', 'wb') as outfile:
    pickle.dump(final_results, outfile)
