import pandas as pd
import ast
from f1 import f1_score_with_precision_recall
import seaborn as sns
import matplotlib.pyplot as plt

def calc_bio_f1_prec_recall(data, model):
    f1_sum_gen = 0
    f1_sum_rag = 0
    prec_sum_gen = 0
    prec_sum_rag = 0
    recall_sum_gen = 0
    recall_sum_rag = 0
    for _, item in data.iterrows():
        if model == 'llama':
            f1_gen = f1_score_with_precision_recall(item['exact_answers'], ast.literal_eval(item['gen_outputs'])[1].lower())
            f1_rag = f1_score_with_precision_recall(item['exact_answers'],
                                                    ast.literal_eval(item['rag_outputs'])[1].lower())
        elif model == 'olmo':
            f1_gen = f1_score_with_precision_recall(item['exact_answers'], ast.literal_eval(item['gen_outputs'])[0].split("assistant<|end_header_id|>\n\n")[-1].lower())
            f1_rag = f1_score_with_precision_recall(item['exact_answers'],
                                                    ast.literal_eval(item['rag_outputs'])[0].split("assistant<|end_header_id|>\n\n")[-1].lower())
        else:
            raise NotImplementedError
        f1_sum_gen+=f1_gen['f1']
        prec_sum_gen += f1_gen['precision']
        recall_sum_gen += f1_gen['recall']

        f1_sum_rag+=f1_rag['f1']
        prec_sum_rag += f1_rag['precision']
        recall_sum_rag += f1_rag['recall']


    gen_f1 = f1_sum_gen / len(data)
    rag_f1 = f1_sum_rag / len(data)

    gen_precision = prec_sum_gen / len(data)
    rag_precision = prec_sum_rag / len(data)

    gen_recall = recall_sum_gen / len(data)
    rag_recall = recall_sum_rag / len(data)

    res = {'gen_f1': gen_f1,
           'rag_f1': rag_f1,
           'gen_precision': gen_precision,
           'rag_precision': rag_precision,
           'gen_recall': gen_recall,
           'rag_recall': rag_recall,
           }
    return res

def calc_trivia_f1_prec_recall(data, model):
    f1_sum_gen = 0
    f1_sum_rag = 0
    prec_sum_gen = 0
    prec_sum_rag = 0
    recall_sum_gen = 0
    recall_sum_rag = 0
    for _, item in data.iterrows():
        answers = ast.literal_eval(item['answer'])
        f1_gen_best = 0
        prec_gen_best = 0
        recall_gen_best = 0
        f1_rag_best = 0
        prec_rag_best = 0
        recall_rag_best = 0
        for answer in answers:
            if model == 'llama':
                f1_gen = f1_score_with_precision_recall(answer, ast.literal_eval(item['gen_decoded_outputs'])[1].lower())
                f1_rag = f1_score_with_precision_recall(answer,
                                                        ast.literal_eval(item['rag_decoded_outputs'])[1].lower())
            elif model == 'olmo':
                f1_gen = f1_score_with_precision_recall(answer,
                                                        ast.literal_eval(item['gen_decoded_outputs'])[0].split("assistant<|end_header_id|>\n\n")[-1].lower())
                f1_rag = f1_score_with_precision_recall(answer,
                                                        ast.literal_eval(item['rag_decoded_outputs'])[0].split("assistant<|end_header_id|>\n\n")[-1].lower())
            else:
                raise NotImplementedError
            if f1_gen['f1'] >f1_gen_best:
                f1_gen_best = f1_gen['f1']
                prec_gen_best = f1_gen['precision']
                recall_gen_best = f1_gen['recall']

            if f1_rag['f1'] >f1_rag_best:
                f1_rag_best = f1_rag['f1']
                prec_rag_best = f1_rag['precision']
                recall_rag_best = f1_rag['recall']

        f1_sum_gen+=f1_gen_best
        prec_sum_gen += prec_gen_best
        recall_sum_gen += recall_gen_best

        f1_sum_rag+=f1_rag_best
        prec_sum_rag += prec_rag_best
        recall_sum_rag += recall_rag_best


    gen_f1 = f1_sum_gen / len(data)
    rag_f1 = f1_sum_rag / len(data)

    gen_precision = prec_sum_gen / len(data)
    rag_precision = prec_sum_rag / len(data)

    gen_recall = recall_sum_gen / len(data)
    rag_recall = recall_sum_rag / len(data)

    res = {'gen_f1': gen_f1,
           'rag_f1': rag_f1,
           'gen_precision': gen_precision,
           'rag_precision': rag_precision,
           'gen_recall': gen_recall,
           'rag_recall': rag_recall,
           }
    return res

print("bio llama")
bio_res = pd.read_csv("bioasq_qa_results.csv")
res_bio_llama =  calc_bio_f1_prec_recall(bio_res, model = "llama")
for key, value in res_bio_llama.items():
    print(key, value)
print("\n")

print("trivia llama")
trivia_res = pd.read_csv("trivia_qa_results_new.csv")
trivia_res_non_freeform = trivia_res.loc[trivia_res['question_type']!='FreeForm']
res_trivia_llama = calc_trivia_f1_prec_recall(trivia_res_non_freeform, model = "llama")
for key, value in res_trivia_llama.items():
    print(key, value)
print("\n")

print("trivia olmo")
trivia_res = pd.read_csv("trivia_qa_results_new_olmo.csv")
trivia_res_non_freeform = trivia_res.loc[trivia_res['question_type']!='FreeForm']
res_trivia_olmo = calc_trivia_f1_prec_recall(trivia_res_non_freeform, model = "olmo")
for key, value in res_trivia_olmo.items():
    print(key, value)
print("\n")

print("bio olmo")
bio_res_olmo = pd.read_csv("bioasq_qa_results_new_olmo.csv")
bio_res_olmo.rename(columns={'gen_decoded_outputs': 'gen_outputs', 'rag_decoded_outputs': 'rag_outputs'}, inplace=True)
res_bio_olmo = calc_bio_f1_prec_recall(bio_res_olmo, model = "olmo")
for key, value in res_bio_olmo.items():
    print(key, value)
print("\n")

print("trivia llama")
trivia_res = pd.read_csv("trivia_qa_results_new.csv")
trivia_res_non_freeform = trivia_res.loc[trivia_res['question_type']!='FreeForm']
res_trivia_llama = calc_trivia_f1_prec_recall(trivia_res_non_freeform, model = "llama")
for key, value in res_trivia_llama.items():
    print(key, value)
print("\n")

avg_f1 = pd.DataFrame(dict(F1=[0.617, 0.720, 0.436, 0.519], Precision=[0.682, 0.791, 0.569, 0.785], Recall=[0.596, 0.701, 0.417, 0.484], QA=["TriviaQA", "TriviaQA", "BioASQ-QA", "BioASQ-QA"], Model=["Generation", "RAG", "Generation", "RAG"]))
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.lineplot(data = avg_f1, ax = axs[0], x = "Model", y = "F1", hue="QA")
axs[0].set_ylim(0, 1)
axs[0].set_title("Average F1")

sns.lineplot(data = avg_f1, ax = axs[1], x = "Model", y = "Precision", hue="QA")
axs[1].set_ylim(0, 1)
axs[1].set_title("Average Precision")

sns.lineplot(data = avg_f1, ax = axs[2], x = "Model", y = "Recall", hue="QA")
plt.tight_layout()
axs[2].set_ylim(0, 1)
axs[2].set_title("Average Recall")

plt.savefig('eval.png', dpi=600)
plt.show()