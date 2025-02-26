import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
import os
import logging
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger("")
logging.basicConfig(filename= f'./logs/entailment-{datetime.now().isoformat()}.log', encoding='utf-8', level=logging.INFO)

def check_deberta_bidirectional(phrase1, phrase2) -> int:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").cuda()

    inputs = tokenizer(phrase1, phrase2, return_tensors="pt").to('cuda')
    # The model checks if text1 -> text2, i.e. if text2 follows from text1.
    # check_implication('The weather is good', 'The weather is good and I like you') --> 1
    # check_implication('The weather is good and I like you', 'The weather is good') --> 2
    outputs = model(**inputs)
    logits = outputs.logits
    # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
    largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
    prediction = largest_index.cpu().item()
    return prediction


def deberta_prompt(question, answer):
    return f'''Question: {question}
Answer: {answer}'''

def get_deberta_entailment(question, phrase1, phrase2, strict=True) -> bool:
    forward = check_deberta_bidirectional(
        deberta_prompt(question, phrase1), deberta_prompt(question, phrase2))
    reverse = check_deberta_bidirectional(
        deberta_prompt(question, phrase2), deberta_prompt(question, phrase1))

    if strict:
            semantically_equivalent = (forward == 2) and (reverse == 2)
    else:
        implications = [forward, reverse]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent

SemanticSet = dict[int, int]

client = OpenAI(
    api_key=os.environ["OPENAI_KEY"],
)


def gpt_entailment_prompt(question, text1, text2):
    prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
    prompt += "Here are two possible answers:\n"
    prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
    prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
    prompt += "Response:"""
    return prompt

def get_llm_entailement_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
        temperature=0.0,
        max_tokens=200,
    )

    binary_response = chat_completion.choices[0].message.content.lower()

    if 'entailment' in binary_response:
        return 2
    elif 'neutral' in binary_response:
        return 1
    elif 'contradiction' in binary_response:
        return 0
    else:
        logging.warning('MANUAL NEUTRAL!')
        logging.warning(prompt)
        logging.warning(binary_response)
        return 1

def get_gpt_entailment(question, text1, text2, strict=True) -> bool:
    forward = get_llm_entailement_response(gpt_entailment_prompt(question, text1, text2))
    reverse = get_llm_entailement_response(gpt_entailment_prompt(question, text2, text1))

    if strict:
            semantically_equivalent = (forward == 2) and (reverse == 2)
    else:
        implications = [forward, reverse]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent

def gpt_oneshot_entailment_prompt(question, texts):
    system = f"""You are evaluating the semantic similarity of answers to a technical question.\n"""
    system += "Please group the question and answers pairs according to their semantic similarity. To evaluate an answer you must ensure:\n"""
    system += "- Each answer is considered within the context of the question.\n"""
    system += "- Syntax, punctuation, or word order do not affect semantics.\n"""
    system += "- Additional information that doesn't change the meaning of an answer does not affect semantics. For example, the answer 'dogs' and 'dogs, such as a terrier' are the the same.\n"""
    system += "- Information in an answer, that is also in the question doesn't affect the semantics of the answer.\n"""
    system += "- Additional clauses that don't change the underlying meaning does not affect semantics. For example, the answers 'X', 'X and Y', 'X and Z' are the same where 'Y' and 'Z' do not change the meaning of those answers.\n"""

    user = "Group the following question and answer pairs according to semantic similarity:\n"
    for idx, text in enumerate(texts):
         user += f"""{idx}) {question}: {text}\n"""
    return system, user


class GroupId(BaseModel):
    id: int

class AnswerIndexToGroupId(BaseModel):
    answer_0: GroupId
    answer_1: GroupId
    answer_2: GroupId
    answer_3: GroupId
    answer_4: GroupId
    answer_5: GroupId
    answer_6: GroupId
    answer_7: GroupId
    answer_8: GroupId
    answer_9: GroupId


def get_oneshot_llm_entailement_response(system, user):
    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            }
        ],
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1280,
        response_format=AnswerIndexToGroupId
    )

    clusters: AnswerIndexToGroupId = chat_completion.choices[0].message.parsed 
    print("straight up", clusters)
    return clusters


def get_oneshot_gpt_entailment(question, answers) -> dict:
    system, user = gpt_oneshot_entailment_prompt(question, answers)
    clusters = get_oneshot_llm_entailement_response(system, user)
    cluster_dict = {}
    for idx in range(10):
        cluster_dict[idx] = clusters.dict()[f"answer_{idx}"]["id"]
    return cluster_dict


def gpt_oneshot_correctness_all(question, true_answer, answers):
    prompt = f"""You are an expert senior obstetrics and gynaecology doctor evaluating the correctness of answers to the following question \"{question}\"\n"""
    prompt += f"The true answer is: {true_answer}\n"
    prompt += "Here are the possible answers:\n"
    for idx, text in enumerate(answers):
         prompt += f"""{idx}. {text}\n"""

    prompt += "Are all possible answers correct with respect to the true answer?\n"""
    return prompt

def gpt_oneshot_correctness_some(question, true_answer, answers):
    prompt = f"""You are an expert senior obstetrics and gynaecology doctor evaluating the correctness of answers to the following question \"{question}\"\n"""
    prompt += f"The true answer is: {true_answer}\n"
    prompt += "Here are the possible answers:\n"
    for idx, text in enumerate(answers):
         prompt += f"""{idx}. {text}\n"""

    prompt += "Are most possible answers correct with respect to the true answer?\n"""
    return prompt

class Correctness(BaseModel):
    correct: bool

def get_oneshot_llm_correctness_response(prompt):
    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1280,
        response_format=Correctness
    )

    correct: Correctness = chat_completion.choices[0].message.parsed 
    return correct.correct

def get_oneshot_gpt_correctness(question, true_answer, answers) -> bool:
    prompt_all = gpt_oneshot_correctness_all(question, true_answer, answers)
    prompt_most = gpt_oneshot_correctness_some(question, true_answer, answers)

    all = get_oneshot_llm_correctness_response(prompt_all)
    most = get_oneshot_llm_correctness_response(prompt_most)
    return all, most

def embedding_preorder_entailment(question, answers):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = [model.encode(answer) for answer in answers]
    dist = cos_sim
    distances = []
    for i in range(len(answers)):
        distance_sum = 0
        for j in range(len(answers)):
            if i != j:
                distance = dist(embeddings[i],embeddings[j])
                distance_sum += distance
        distances.append(distance_sum)
    ans_dist = list(zip(answers, distances))
    ans_dist.sort(key=lambda x: x[1])
    answers = [x[0] for x in ans_dist]

    semantic_set_ids= {}
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

    return semantic_set_ids




