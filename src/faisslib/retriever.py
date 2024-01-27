import os
import lmdb
import json
import time
import faiss
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
logger = logging.getLogger(__name__)

class Retriever(object):
    def __init__(self, retriever_dir, nprobe, topk=1, device_id=-1, index_path=None):
        # Load metadata
        with open(os.path.join(retriever_dir, 'metadata.json'), 'r') as fin:
            self.metadata = json.load(fin)
        # Create the database
        db_path = self.metadata['db_path'] if 'db_path' in self.metadata else os.path.join(retriever_dir, 'db')
        self.env = lmdb.open(db_path, readonly=True, readahead=True)
        # The index path is not given, use the default index path
        if index_path == None:
            if 'index_path' in self.metadata: # Use the index path in the metadata
                index_path = self.metadata['index_path']
            else: # Use the default index path
                index_dir = os.path.join(retriever_dir, 'index')
                index_path = glob(os.path.join(index_dir, '*.index'))[0]
        # Create the index
        self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        if device_id >= 0:
            # Move the index to GPU
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(gpu_res, device_id, self.index, co)
        # Set hyper parameters
        self.index.nprobe = nprobe
        self.topk = topk
        self.retrieval_dim = self.metadata['emb_dim']
        self.cache = None

    def __del__(self):
        self.env.close()

    def search(self, query_embs, post_process_func=None):
        # Search the index
        distances, ids = self.index.search(query_embs, self.topk)
        # Retrieve data from the database
        all_neighbors = {}
        txn = self.env.begin()
        for query_i in range(query_embs.shape[0]):
            neighbors = {}
            for neighbor_j in range(self.topk):
                if ids[query_i][neighbor_j] != -1: # The neighbor is valid
                    # Encode the database key
                    key = txn.get(str(ids[query_i][neighbor_j]).encode())
                    assert key != None, f'Cannot find key {ids[query_i][neighbor_j]}'
                    # Get the database value
                    value = pickle.loads(key)
                    if post_process_func != None: # Use the customized function to post-process the value
                        neighbors = post_process_func(value, neighbors)
                    else: # Use the default processing method
                        text = value['text'] if 'text' in value else None
                        emb = np.expand_dims(np.array(value['embedding']).squeeze(), axis=0)
                        if 'text' not in neighbors:
                            neighbors['text'] = [text]
                            neighbors['emb'] = [emb]
                        else:
                            neighbors['text'].append(text)
                            neighbors['emb'].append(emb)
            # Concate neighbor embeddings for matrix operations
            if 'emb' in neighbors:
                for _ in range(self.topk - len(neighbors['emb'])):
                    neighbors['emb'].append(np.zeros((1, self.retrieval_dim)))
                neighbors['emb'] = np.expand_dims(np.concatenate(neighbors['emb'], axis=0), axis=0)
            # Store neighbors of the i-th query
            all_neighbors[query_i] = neighbors
            return all_neighbors
    
    def save_in_cache(self, neighbors):
        self.cache = neighbors
    
    def fetch_from_cache(self):
        return self.cache

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_dir', type=str, default=None)
    parser.add_argument('--nprobe', type=int, default=500)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    retriever = Retriever(args.retriever_dir, args.nprobe, topk=args.topk, device_id=args.device_id)

    # Test the retrieval quality
    emb_files = glob(os.path.join(args.retriever_dir, 'emb/*.json.gz'), recursive=True)
    emb_files.sort(key=lambda emb_file: int(emb_file.split('/')[-1].removesuffix('.json.gz').split('-')[-1]))
    for emb_file in emb_files:
        # Load embeddings
        df = pd.read_json(emb_file, orient='records', lines=True)
        embeddings = np.array([emb for emb in df['embedding'].values])

        results = {}
        start = time.time_ns()
        num_batches = int(np.ceil(embeddings.shape[0] / args.batch_size))
        for batch_id in range(num_batches):
            l = batch_id * args.batch_size
            r = (batch_id + 1) * args.batch_size
            neighbors = retriever.search(embeddings[l:r])
            results = results | neighbors
        end = time.time_ns()
        latency = (end - start) / (num_batches * args.batch_size)
        logger.info(f'Average latency {latency:.2f} ns')

        recall = 0
        all_embeddings = np.concatenate([results[query_i]['emb'] for query_i in sorted(results)], axis=0)
        distance = np.ones(embeddings.shape[0])
        for i in range(args.topk):
            distance *= (all_embeddings[:, i, :] - embeddings).sum(axis=1)
        recall = (distance == 0).sum()
        logger.info(f'Recall@{args.topk} {recall / embeddings.shape[0]}')
