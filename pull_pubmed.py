import json
from Bio import Entrez
from tqdm import tqdm
import time
import re
import csv

Entrez.email = "alexandrasalem8@gmail.com"

with open('7655130/training11b.json', 'r') as f:
    data = json.load(f)

all_ids = []
for item in data['questions']:
    docs = item['documents']
    ids = [doc.split('/')[-1] for doc in docs]
    all_ids.extend(ids)

all_ids = list(set(all_ids))


def fetch_abstracts(pmids, batch_size=200):
    abstracts = {}
    for start in tqdm(range(0, len(pmids), batch_size)):
        end = min(start + batch_size, len(pmids))
        batch = pmids[start:end]
        #print(f"Fetching PMIDs {start+1} to {end}...")

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="text",
            )
            data = handle.read()
            handle.close()
            data_split = re.split('\n\d+\. |^\d+\. ', data)[1:]
            abstracts.update({pmid: entry.strip() for pmid, entry in zip(batch, data_split)})
        except Exception as e:
            print(e)
            print(f"Error fetching PMIDs {batch}: {e}")
            time.sleep(5)

        time.sleep(0.4)  # respect NCBI rate limits
    return abstracts



abstracts = fetch_abstracts(pmids=all_ids)

with open('pubmed_abstracts.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'abstract']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for key, value in abstracts.items():
       writer.writerow([key, value])

