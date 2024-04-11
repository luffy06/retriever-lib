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
from tqdm import tqdm
from glob import glob
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class FaissRetriever(object):
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
        assert type(query_embs) == np.ndarray, f'query embeddings should be np.ndarray type'
        assert len(query_embs.shape) == 2, f'query embeddings should be 2D arrays'
        if query_embs.dtype != np.dtype('float32'):
            logger.warn(f'Changing the data type from {query_embs.dtype} to float32')
            query_embs = query_embs.astype('float32')
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
                        if 'text' not in neighbors:
                            neighbors['text'] = [text]
                        else:
                            neighbors['text'].append(text)
                        if 'embedding' in value:
                            emb = np.expand_dims(np.array(value['embedding']).squeeze(), axis=0)
                            if 'emb' not in neighbors:
                                neighbors['emb'] = [emb]
                            else:
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
    parser.add_argument('--nprobe', type=int, default=512)
    parser.add_argument('--device_id', type=int, default=-1)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    logger.info(f'Parameters {args}')

    retriever = FaissRetriever(
        retriever_dir=args.retriever_dir, 
        nprobe=args.nprobe, 
        topk=args.topk, 
        device_id=args.device_id
    )
    logger.info(f'Finish loading the retriever')

    # Test the retrieval quality
    emb_files = glob(os.path.join(args.retriever_dir, 'emb/*.json.gz'), recursive=True)
    emb_files.sort(key=lambda emb_file: int(emb_file.split('/')[-1].removesuffix('.json.gz').split('-')[-1]))
    for i, emb_file in enumerate(emb_files):
        # Load embeddings
        df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
        embeddings = np.array([emb for emb in df['embedding'].values]).astype('float32')
        logger.info(f'Load {embeddings.shape[0]} embeddings {embeddings.shape} for evaluation')

        # Evaluate the latency
        results = {}
        start = time.time_ns()
        num_batches = int(np.ceil(embeddings.shape[0] / args.batch_size))
        for batch_id in tqdm(range(num_batches)):
            l = batch_id * args.batch_size
            r = (batch_id + 1) * args.batch_size
            neighbors = retriever.search(embeddings[l:r, :])
            results = results | neighbors
        latency = (time.time_ns() - start) / embeddings.shape[0]
        logger.info(f'Average latency {latency:.2f} ns')

        # Evaluate the recall
        if 'emb' in results[0]:
            recall = 0
            all_embeddings = np.concatenate([results[query_i]['emb'] for query_i in sorted(results)], axis=0)
            distance = np.ones(embeddings.shape[0])
            for i in range(args.topk):
                distance *= (all_embeddings[:, i, :] - embeddings).sum(axis=1)
            recall = (distance == 0).sum()
            logger.info(f'Recall@{args.topk} {recall / embeddings.shape[0]}')
