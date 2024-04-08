import os
import lmdb
import json
import faiss
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def get_faiss_metric_type(metric_type):
    if metric_type == 'inner_product':
        return faiss.METRIC_INNER_PRODUCT
    elif metric_type == 'L1':
        return faiss.METRIC_L1
    return faiss.METRIC_L2

def get_name(size):
    if size < 1000:
        return f'{size}'
    elif size < 1000000:
        return f'{size // 1000}k'
    else:
        return f'{size // 1000000}m'

class FaissRetrieverBuilder(object):
    def __init__(
        self, 
        model_path, 
        data_dir=None, 
        output_dir='metadata', 
        num_chunks_per_file=1000, 
        batch_size=32,
        device_id=-1, 
        do_chunk=True,
        do_encode=True,
    ):
        self.output_dir = output_dir
        self.split_dir = os.path.join(self.output_dir, 'splits')
        self.emb_dir = os.path.join(self.output_dir, 'emb')
        self.db_base_dir = os.path.join(self.output_dir, 'db')
        self.index_base_dir = os.path.join(self.output_dir, 'index')
        self.meta_path = os.path.join(self.output_dir, 'metadata.json')
        self.device_id = device_id
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if do_chunk:
            assert data_dir != None
            self._chunk(data_dir, num_chunks_per_file)
        if do_encode:
            if self.device_id >= 0:
                model = SentenceTransformer(model_path, device=f'cuda:{self.device_id}')
            else:
                model = SentenceTransformer(model_path, device=f'cpu')
            self._encode(model, batch_size=batch_size)
            del model
            with open(self.meta_path, 'w') as fout:
                json.dump(self.metadata, fout)
        assert os.path.exists(self.meta_path)
        with open(self.meta_path, 'r') as fin:
            self.metadata = json.load(fin)
        logger.info(f'Embedding Information {self.metadata}')

    def __get_source_data_files(self, data_dir):
        data_files = glob(os.path.join(data_dir, '*.json'), recursive=True)
        data_files.sort()
        return data_files

    def __get_split_files(self):
        split_files = glob(os.path.join(self.split_dir, '*.json'), recursive=True)
        split_files.sort()
        return split_files

    def __get_embedding_files(self):
        emb_files = glob(os.path.join(self.emb_dir, '*.json.gz'), recursive=True)
        emb_files.sort(key=lambda emb_file: int(emb_file.split('/')[-1].removesuffix('.json.gz').split('-')[-1]))
        return emb_files

    def __chunk_by_sentence(self, document, end_words=['.', '! ']):
        chunks = []
        chunk = ''
        for i, w in enumerate(document):
            chunk += w
            if any([w.endswith(sw) for sw in end_words]):
                if i != len(document) - 1 \
                    and (document[i + 1] == ' ' or document[i + 1] == '\n'):
                    chunks.append(chunk.strip())
                    chunk = ''
                elif i == len(document) -1:
                    chunks.append(chunk.strip())
                    chunk = ''
        if chunk != '':
            chunks.append(chunk.strip())
        return list(filter(lambda x: x.strip() != '', chunks))

    def _chunk(self, data_dir, num_chunks_per_file=1000, chunking_strategy='sentence'):
        logger.info(f'Split the chunk in {data_dir}, the number of chunks in each split is about {num_chunks_per_file}')
        if not os.path.exists(self.split_dir):
            os.mkdir(self.split_dir)
        
        data_files = self.__get_source_data_files(data_dir)
        for data_path in data_files:
            num_splits = 0
            all_chunks = []
            basename = data_path.split('/')[-1].removesuffix('.json')
            logger.info(f'Split chunks in {basename}')
            with open(data_path, 'r') as fin:
                for line in fin:
                    # Load formatted data
                    line = line.strip()
                    try:
                        data = json.loads(line)
                    except:
                        raise Exception(f'Not a json-formatted row {line}')
                    # Chunk the source text
                    if chunking_strategy == 'sentence':
                        chunks = self.__chunk_by_sentence(data['text'])
                    else:
                        raise Exception(f'Unknown chunking strategy {chunking_strategy}')
                    # Collect and save chunks
                    if len(all_chunks) + len(chunks) > num_chunks_per_file:
                        num_trunc = num_chunks_per_file - len(all_chunks)
                        all_chunks.extend(chunks[:num_trunc])
                        df = pd.DataFrame({'text': all_chunks})
                        split_path = os.path.join(self.split_dir, f'{basename}-{num_splits}.json')
                        df.to_json(split_path, orient='records', lines=True)
                        num_splits += 1
                        all_chunks = chunks[num_trunc:]
                    else:
                        all_chunks.extend(chunks)
            # Save the last chunk
            if len(all_chunks) > 0:
                df = pd.DataFrame({'text': all_chunks})
                split_path = os.path.join(self.split_dir, f'{basename}-{num_splits}.json')
                df.to_json(split_path, orient='records', lines=True)
                num_splits += 1

    def _encode(self, model, batch_size=32):
        logger.info(f'Encode the data in {self.split_dir}')
        if not os.path.exists(self.emb_dir):
            os.mkdir(self.emb_dir)
        
        self.metadata = {'num_emb': 0, 'emb_dim': 0}
        split_files = self.__get_split_files()
        for i, split_file in enumerate(split_files):
            basename = split_file.split('/')[-1].removesuffix('.json')
            logger.info(f'Encode the split file [{split_file}] ({i + 1}/{len(split_files)})')
            # Load chunks
            df = pd.read_json(split_file, orient='records', lines=True)
            chunks = df['text'].values
            # Encoding
            embeddings = model.encode(chunks, batch_size=batch_size)
            # Update metadata
            self.metadata['num_emb'] += embeddings.shape[0]
            self.metadata['emb_dim'] = embeddings.shape[1]
            # Save embeddings
            df.insert(0, 'embedding', embeddings.tolist())
            emb_path = os.path.join(self.emb_dir, f'{basename}.json.gz')
            df.to_json(emb_path, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
            del embeddings

    def build(
        self, 
        build_index,
        build_db,
        index_type,
        map_size=200*1024*1024*1024,
        build_size=None,
        metric_type='L2',
        max_train_size=None,
        train_ratio=0.1, 
        least_num_train=1000000, 
        sub_index_size=1000000
    ):
        if build_size != None:
            dirname = get_name(build_size)
        else:
            dirname = 'all'
        self.db_dir = os.path.join(self.output_dir, os.path.join(dirname, 'db'))
        self.index_dir = os.path.join(self.output_dir, os.path.join(dirname, 'index'))
        self.meta_path = os.path.join(self.output_dir, os.path.join(dirname, 'metadata.json'))
        if build_size == None:
            self.build_size = self.metadata['num_emb']
        else:
            self.build_size = np.minimum(build_size, self.metadata['num_emb'])
        logger.info(f'{self.build_size} data are used to build the db and index')
        if build_db:
            self._build_db(map_size=map_size)
        if build_index:
            self._build_index(
                index_type=index_type, 
                metric_type=metric_type,
                max_train_size=max_train_size,
                train_ratio=train_ratio, 
                least_num_train=least_num_train, 
                sub_index_size=sub_index_size
            )
        if not os.path.exists(self.meta_path):
            with open(self.meta_path, 'w') as fout:
                json.dump(self.metadata, fout)
    
    def _build_db(self, map_size=200*1024*1024*1024):
        logger.info(f'Build the database')
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        # Collect the statistics of existing database
        env = lmdb.open(self.db_dir, map_size=map_size)
        db_size = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, __ in cursor:
                db_size += 1
        logger.info(f'Database initial size: {db_size}')
        if db_size != 0:
            # Clear the existing database
            logger.info(f'Clear database')
            with env.begin(write=True) as txn:
                cursor = txn.cursor()
                for key, __ in cursor:
                    txn.delete(key)
            db_size = 0

        last_row = None
        emb_files = self.__get_embedding_files()
        for i, emb_file in enumerate(emb_files):
            # Load embeddings
            df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
            # Store data to the database
            txn = env.begin(write=True)
            for j, row in df.iterrows():
                txn.put(
                    str(db_size).encode(),
                    pickle.dumps({
                        'text': row['text'],
                        'embedding': row['embedding']
                    })
                )
                db_size += 1
                if db_size >= self.build_size:
                    break
            txn.commit()
            logger.info(f'Build the db ({db_size}/{self.build_size})')
            if db_size >= self.build_size:
                break
        env.close()
        logger.info(f'Finish builing the database, size {db_size}')

    def __sample_train_data(self, max_train_size, train_ratio, least_num_train):
        if max_train_size != None:
            num_train = np.min((max_train_size, self.build_size))
        else:
            # Compute the number of data for training the index
            num_train = int(self.build_size * train_ratio)
            num_train = np.max((num_train, np.min((self.build_size, least_num_train))))
        logger.info(f'Sample {num_train} data for training')
        # Create training index across multiple embedding files
        train_idx = np.arange(self.build_size)
        np.random.shuffle(train_idx)
        train_idx = train_idx[:num_train]
        train_idx.sort()
        

        train_embs = []
        idx_count = 0
        emb_files = self.__get_embedding_files()
        for emb_file in emb_files:
            # Load embeddings
            df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
            embeddings = np.array([d for d in df['embedding'].values]).astype('float32')
            # Select embeddings based on the training index
            start_idx = idx_count
            end_idx = start_idx + embeddings.shape[0]
            idx = train_idx
            idx = idx[idx >= start_idx]
            idx = idx[idx < end_idx]
            # Collect training embeddings
            sub_embs = np.take(embeddings, idx - idx_count, axis=0)
            train_embs.append(sub_embs)
            idx_count += embeddings.shape[0]
            assert self.metadata['emb_dim'] == embeddings.shape[1]
            del embeddings
            logger.info(f'Sampling training data ({idx_count}/{self.build_size})')
            if idx_count >= self.build_size:
                break
        # Concat all training embeddings
        train_embs = np.concatenate(train_embs, axis=0).astype('float32')
        assert train_embs.shape[0] == num_train, f'sample {train_embs.shape[0]} training data but {num_train} data are required'
        assert train_embs.shape[1] == self.metadata['emb_dim']
        return train_embs

    def __train_index(self, train_embs, index_type, metric_type, trained_index_path):
        # Create the index
        index = faiss.index_factory(train_embs.shape[1], index_type, get_faiss_metric_type(metric_type))
        if self.device_id >= 0:
            # Move the index to GPU
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(gpu_res, self.device_id, index, co)
        # Train the index
        logger.info(f'Train the index on {train_embs.shape[0]} embeddings')
        index.train(train_embs)
        if self.device_id >= 0:
            index = faiss.index_gpu_to_cpu(index)
        # Store the index
        faiss.write_index(index, trained_index_path)

    def __build_sub_index(self, buffer_embs, trained_index_path, sub_index_path):
        # Load the trained index
        index = faiss.read_index(trained_index_path)
        if self.device_id >= 0:
            # Move the index to GPU
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(gpu_res, self.device_id, index, co)
        # Add data to the existing index
        index.add(buffer_embs)
        if self.device_id >= 0:
            index = faiss.index_gpu_to_cpu(index)
        # Store the index
        faiss.write_index(index, sub_index_path)

    def _build_index(
        self, 
        index_type, 
        metric_type,
        train_ratio, 
        least_num_train, 
        sub_index_size
    ):
        logger.info(f'Build the index [{index_type}]')
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        # Create the path for index
        index_path_prefix = index_type.lower().replace(',', '-').replace('_', '-')
        index_path_prefix = index_path_prefix + '-' + metric_type.lower().replace(',', '-').replace('_', '-')
        index_path_prefix = os.path.join(self.index_dir, index_path_prefix)
        trained_index_path = index_path_prefix + '.trained'
        # Sample training data
        train_embs = self.__sample_train_data(train_ratio, least_num_train)
        # Train the index
        self.__train_index(train_embs, index_type, metric_type, trained_index_path)

        logger.info(f'Start to build the sub indexes')
        # TODO: When not building with all data, the data need to be randomized for generability
        rest_to_build = self.build_size
        num_sub_indexes = 0
        buffer_embs = []
        buffer_size = 0
        emb_files = self.__get_embedding_files()
        for i, emb_file in enumerate(emb_files):
            # Load embeddings
            df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
            embeddings = np.array([d for d in df['embedding'].values]).squeeze().astype('float32')
            # Truncate the embedding according to the building size
            if rest_to_build > embeddings.shape[0]:
                rest_to_build -= embeddings.shape[0]
                logger.info(f'Build the index ({self.build_size - rest_to_build}/{self.build_size})')
            elif rest_to_build > 0:
                embeddings = embeddings[:rest_to_build, :]
                rest_to_build = 0
                logger.info(f'Build the index ({self.build_size - rest_to_build}/{self.build_size})')
            else:
                break
            # Buffer the embeddings for batch-wise building
            buffer_embs.append(embeddings)
            buffer_size += embeddings.shape[0]
            # Build the sub index
            if buffer_size > sub_index_size or i == len(emb_files) - 1:
                # Concat the buffer embeddings
                buffer_embs = np.concatenate(buffer_embs, axis=0)
                assert buffer_embs.shape[0] == buffer_size
                assert buffer_embs.shape[1] == self.metadata['emb_dim']
                # Create the path for the sub index
                sub_index_path = index_path_prefix + f'-sub{num_sub_indexes}.index'
                # Build the sub index
                self.__build_sub_index(buffer_embs, trained_index_path, sub_index_path)
                del buffer_embs
                # Update metadata
                num_sub_indexes = num_sub_indexes + 1
                buffer_embs = []
                buffer_size = 0
        # End loop of building sub indexes, and start to merge all sub indexes
        logger.info(f'Merge all sub indexes')
        # Load the trained index
        trained_index = faiss.read_index(trained_index_path)
        for i in tqdm(range(num_sub_indexes)):
            # Merge the i-th sub index with the trained index
            sub_index_path = index_path_prefix + f'-sub{i}.index'
            sub_index = faiss.read_index(sub_index_path)
            trained_index.merge_from(sub_index, trained_index.ntotal)
            del sub_index
        # Store the final index
        index_path = index_path_prefix + f'.index'
        faiss.write_index(trained_index, index_path)
        logger.info(f'Finish building indexes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='metadata')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--do_chunk', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_chunks_per_file', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--do_encode', action=argparse.BooleanOptionalAction)
    parser.add_argument('--build_size', type=int, default=None)
    parser.add_argument('--build_db', action=argparse.BooleanOptionalAction)
    parser.add_argument('--build_index', action=argparse.BooleanOptionalAction)
    parser.add_argument('--map_size', type=int, default=200*1024*1024*1024)
    parser.add_argument('--max_train_size', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--least_num_train', type=int, default=1000)
    parser.add_argument('--index_type', type=str, default='IVF65536_HNSW32,PQ64')
    parser.add_argument('--metric_type', type=str, default='L2')
    parser.add_argument('--sub_index_size', type=int, default=10000)
    args = parser.parse_args()
    logger.info(f'Parameters {args}')

    builder = FaissRetrieverBuilder(
        args.model_path, 
        data_dir=args.data_dir, 
        output_dir=args.output_dir, 
        num_chunks_per_file=args.num_chunks_per_file, 
        batch_size=args.batch_size,
        device_id=args.device_id,
        do_chunk=args.do_chunk,
        do_encode=args.do_encode,
    )
    builder.build(
        args.build_index,
        args.build_db,
        index_type=args.index_type, 
        map_size=args.map_size,
        build_size=args.build_size,
        metric_type=args.metric_type,
        max_train_size=args.max_train_size,
        train_ratio=args.train_ratio,
        least_num_train=args.least_num_train,
        sub_index_size=args.sub_index_size
    )
