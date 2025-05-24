import json
import pandas as pd
from ir import ir_single_query_top_docs_trivia, ir_single_query_top_docs_bio
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)
model.eval()

def run_model_on_trivia_qa(data_location):
    with open(data_location, 'r') as f:
        data = json.load(f)

    questions = []
    question_ids = []
    answers = []
    question_types = []
    answer_locations = []
    retrieved_ids = []
    gen_prompts = []
    rag_prompts = []

    data = data['Data']
    for item in tqdm(data):
        question = item['Question']
        question_id = item['QuestionId']
        answer = item['Answer']['Aliases']
        question_type = item['Answer']['Type']
        answer_location = [source['Filename'] for source in item['EntityPages']]

        questions.append(question)
        question_ids.append(question_id)
        answers.append(answer)
        question_types.append(question_type)
        answer_locations.append(answer_location)

        gen_prompt = "Provide a factual answer to this question using between 1 and 5 words."

        docs = ir_single_query_top_docs_trivia(question, corpus_location='triviaqa-rc/evidence/wikipedia', k=1)
        docs_str = ""
        for item in docs['text']:
            docs_str += item + "\n\n"
        rag_prompt = f"Here are some relevant articles: {docs_str}\n\n{gen_prompt}"
        retrieved_id = list(docs['id'])

        gen_messages = [
            {"role": "system", "content": gen_prompt},
            {"role": "user", "content": question},
        ]
        rag_messages = [
            {"role": "system", "content": rag_prompt},
            {"role": "user", "content": question},
        ]
        gen_prompt_string = tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
        rag_prompt_string = tokenizer.apply_chat_template(rag_messages, tokenize=False, add_generation_prompt=True)


        gen_prompts.append(gen_prompt_string)
        rag_prompts.append(rag_prompt_string)
        retrieved_ids.append(retrieved_id)
    print("done with tfidf")
    new_results = {'question': questions, 'question_id': question_ids, 'answer': answers,
                       'question_type': question_types, 'answer_location': answer_locations,
                       'retrieved_id': retrieved_ids, 'gen_prompts': gen_prompts,
                   'rag_prompts': rag_prompts}
    pd.DataFrame(new_results).to_csv('trivia_qa_ir_results_new.csv', index=False)

    res = pd.read_csv("trivia_qa_ir_results.csv")
    for _, item in res.iterrows():
        gen_prompts = item['gen_prompts']
        rag_prompts = item['rag_prompts']
        item['gen_prompts'] = gen_prompts
        item['rag_prompts'] = rag_prompts
    gen_outputs = []
    rag_outputs = []
    batch_size = 2
    for i in tqdm(range(0, len(res), batch_size)):
        gen_inputs = tokenizer(list(res['gen_prompts'])[i:i + batch_size], padding=True, return_tensors="pt").to(device)
        rag_inputs = tokenizer(list(res['rag_prompts'])[i:i + batch_size], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_output = model.generate(
                **gen_inputs,
                max_new_tokens=512,
            )
            rag_output = model.generate(
                **rag_inputs,
                max_new_tokens=512,
            )
        gen_decoded_outputs = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
        rag_decoded_outputs = tokenizer.batch_decode(rag_output, skip_special_tokens=True)

        gen_decoded_outputs = [out.split("assistant\n\n") for out in gen_decoded_outputs]
        rag_decoded_outputs = [out.split("assistant\n\n") for out in rag_decoded_outputs]
        gen_outputs.extend(gen_decoded_outputs)
        rag_outputs.extend(rag_decoded_outputs)

    res['gen_decoded_outputs'] = gen_outputs
    res['rag_decoded_outputs'] = rag_outputs
    res.to_csv('trivia_qa_results.csv', index=False)

def run_model_on_bio_qa(data_location):
    with open(data_location, 'r') as f:
        data = json.load(f)

    item_types = []
    questions = []
    supporting_doc_ids = []
    exact_answers = []
    item_ids = []
    retrieved_ids = []
    gen_prompts = []
    rag_prompts = []
    print("starting tfidf")
    for item in tqdm(data['questions']):
        item_type = item['type']
        if item_type not in ['factoid', 'yesno']:
            continue
        question = item['body']
        exact_answer = item['exact_answer']
        supporting_docs = item['documents']
        supporting_ids = [doc.split('/')[-1] for doc in supporting_docs]
        item_id = item['id']

        item_types.append(item_type)
        questions.append(question)
        supporting_doc_ids.append(supporting_ids)
        exact_answers.append(exact_answer)
        item_ids.append(item_id)

        if item_type == 'factoid':
            gen_prompt = "Provide a factual answer to this question using between 1 and 5 words."
        elif item_type == 'yesno':
            gen_prompt = "Answer this question with a single word response: either 'yes' or 'no'."
        else:
            raise ValueError("Unknown item type '%s'" % item_type)

        docs = ir_single_query_top_docs_bio(question, corpus_csv="pubmed_abstracts.csv", k=5)
        docs_str = ""
        for item in docs['abstract']:
            docs_str += item + "\n\n"
        rag_prompt = f"Here are some relevant abstracts from PubMed: {docs_str}\n\n{gen_prompt}"
        retrieved_id = list(docs['id'])

        gen_messages = [
            {"role": "system", "content": gen_prompt},
            {"role": "user", "content": question},
        ]
        rag_messages = [
            {"role": "system", "content": rag_prompt},
            {"role": "user", "content": question},
        ]
        gen_prompt_string = tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
        rag_prompt_string = tokenizer.apply_chat_template(rag_messages, tokenize=False, add_generation_prompt=True)


        gen_prompts.append(gen_prompt_string)
        rag_prompts.append(rag_prompt_string)
        retrieved_ids.append(retrieved_id)
    print("done with tfidf")
    new_results = {'item_ids': item_ids, 'item_types': item_types,
                   'questions': questions, 'supporting_doc_ids': supporting_doc_ids,
                   'exact_answers': exact_answers, 'retrieved_ids': retrieved_ids,
                   'gen_prompts': gen_prompts, 'rag_prompts': rag_prompts}
    pd.DataFrame(new_results).to_csv('bioasq_qa_ir_results.csv', index=False)

    res = pd.read_csv("bioasq_qa_ir_results.csv")
    for _, item in res.iterrows():
        gen_prompts = item['gen_prompts']
        rag_prompts = item['rag_prompts']
        gen_prompts = gen_prompts.replace('Provide a brief factual answer to this question.', 'Provide a factual answer to this question using between 1 and 5 words.')
        rag_prompts = rag_prompts.replace('Provide a brief factual answer to this question.', 'Provide a factual answer to this question using between 1 and 5 words.')
        item['gen_prompts'] = gen_prompts
        item['rag_prompts'] = rag_prompts
    gen_outputs = []
    rag_outputs = []
    batch_size = 2
    for i in tqdm(range(0, len(res), batch_size)):
        # add batching
        gen_inputs = tokenizer(list(res['gen_prompts'])[i:i + batch_size], padding=True, return_tensors="pt").to(device)
        rag_inputs = tokenizer(list(res['rag_prompts'])[i:i + batch_size], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_output = model.generate(
                **gen_inputs,
                max_new_tokens=512,
            )
            rag_output = model.generate(
                **rag_inputs,
                max_new_tokens=512,
            )
        gen_decoded_outputs = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
        rag_decoded_outputs = tokenizer.batch_decode(rag_output, skip_special_tokens=True)

        gen_decoded_outputs = [out.split("assistant\n\n") for out in gen_decoded_outputs]
        rag_decoded_outputs = [out.split("assistant\n\n") for out in rag_decoded_outputs]
        gen_outputs.extend(gen_decoded_outputs)
        rag_outputs.extend(rag_decoded_outputs)

    res['gen_decoded_outputs'] = gen_outputs
    res['rag_decoded_outputs'] = rag_outputs
    res.to_csv('bioasq_qa_results.csv', index=False)



run_model_on_bio_qa('7655130/training11b.json')
run_model_on_trivia_qa('triviaqa-rc/qa/verified-wikipedia-dev.json')

