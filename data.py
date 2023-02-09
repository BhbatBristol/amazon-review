import pandas as pd
import json
import gzip
import dgl
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from functools import partial
from tqdm.auto import tqdm
import torch.nn as nn
import dgl.nn.pytorch as dglnn

tqdm.pandas()


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
    df = {}
    i = 0
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def get_embedding(df):
    def _embed_text(text: str, tokenizer, model) -> np.ndarray:
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v for k, v in inputs.items()}
            result = model(**inputs)
            return result.pooler_output[0].cpu().numpy()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    embed = partial(_embed_text, tokenizer=tokenizer, model=model)
    text_embs = df.reviewText.progress_apply(embed)
    df["review_emb"] = text_embs
    return df

def reset_idx(list):
    unique_elements = set(list)
    sorted_elements = sorted(unique_elements)
    indices = {element: i for i, element in enumerate(sorted_elements)}
    y = [indices[element] for element in list]
    return y, dict(map(reversed, indices.items()))

def get_nodes(list1, list2):
    src, src_indices = reset_idx(list1)
    tgt, tgt_indices = reset_idx(list2)
    tgt_tensor = torch.tensor(np.array(tgt))
    src_tensor = torch.tensor(np.array(src))

    return src_tensor, tgt_tensor, src_indices, tgt_indices

def get_tensor(df, indices):

    idx = [indices[i] for i in range(len(indices))]

    np_array = np.stack([df[item] for item in idx])
    if len(np_array.shape) == 1:
        np_array = np.expand_dims(np_array, axis=1)
    return torch.from_numpy(np_array)

def build_graph(df):
    graph_data = dict()
    users, reviews, users_indices, reviews_indices = get_nodes(df.reviewerID.tolist(), df.index.tolist())
    reviews, products, _, products_indices = get_nodes(df.index.tolist(), df.asin.tolist())

    graph_data['user', 'write', 'review'] = (users, reviews)
    graph_data['review', 'writen_by', 'user'] = (reviews, users)
    graph_data['review', 'review_to', 'product'] = (reviews, products)
    graph_data['product', 'reviewed_by', 'review'] = (products, reviews)

    graph = dgl.heterograph(graph_data)

    graph.nodes['review'].data['feat'] = get_tensor(df['review_emb'], reviews_indices).long()
    df.vote = df.vote.fillna('0')
    df.vote[df.vote != '0'] = '1'
    label = get_tensor(df['vote'].astype(float), reviews_indices).squeeze()

    graph.nodes['review'].data['label'] = label.long()


    graph.nodes['user'].data['feat'] = torch.ones(graph.num_nodes('user'), 1).long()
    graph.nodes['product'].data['feat'] = torch.ones(graph.num_nodes('product'), 1).long()


    return graph



def get_graph():
    df = getDF('Appliances_5.json.gz')
    df.reviewText = df.reviewText.fillna('')
    df = get_embedding(df)
    df.to_pickle('dataframe.pkl')
    df = pd.read_pickle('dataframe.pkl')
    graph = build_graph(df)
    print(graph)

    return graph
