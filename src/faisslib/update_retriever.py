import os
import lmdb
import pickle
import argparse
import logging
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

BERT_MODELS = ['bert-base-uncased', 'bert-large-uncased', 'robert-base', 'roberta-large']

def get_embeddings(model_name, embeddings):
    if model_name in BERT_MODELS:
        return embeddings[:, 0, :]
    else:
        return embeddings[:, -1, :]

def update_retriever(path, new_path, model_name, batch_size=32, device='cpu'):    
    # Open the LMDB environment
    env_old = lmdb.open(os.path.join(path, 'db'), readonly=False)
    # Open the new LMDB environment
    if not os.path.exists(os.path.join(new_path, 'db')):
        os.makedirs(os.path.join(new_path, 'db'))
    env_new = lmdb.open(os.path.join(new_path, 'db'), readonly=False, map_size=1099511627776)

    # Load the model based on the provided model_name
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)

    # Begin a write transaction
    txn_old = env_old.begin(write=False)
    txn_new = env_new.begin(write=True)

    batch = {"keys": [], "values": []}

    # Iterate over all the keys in the LMDB
    cursor = txn_old.cursor()
    idx = 0
    for key, value in tqdm(cursor, total=txn_old.stat()['entries']):
        # Deserialize the value (assuming it's stored as a pickled object)
        value = pickle.loads(value)
        batch["keys"].append(key)
        batch["values"].append(value)
        if len(batch["keys"]) == batch_size or idx == txn_old.stat()['entries'] - 1:
            texts = [value['text'] for value in batch["values"]]
            # Generate a new embedding using the last hidden states of the model
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            new_embeddings = get_embeddings(model_name.split('/')[-1], last_hidden_states.detach().cpu().numpy())
            # Update the values with the new embeddings
            for i, key in enumerate(batch["keys"]):
                batch["values"][i]['embedding'] = new_embeddings[i]
                new_value = pickle.dumps(batch["values"][i])
                # Store the updated value back into the LMDB
                txn_new.put(key, new_value)
            batch = {"keys": [], "values": []}
            del inputs, outputs, last_hidden_states, new_embeddings
        idx += 1

    # Close the LMDB environment
    env_old.close()
    env_new.close()

    # if not os.path.exists(os.path.join(new_path, 'index')):
    #     os.makedirs(os.path.join(new_path, 'index'))

    # # Copy the index file from the source path to the target path
    # os.system(f'cp {path}/index/* {new_path}/index')

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, help="Path to the source retriever", required=True)
    parser.add_argument("--target_path", type=str, help="Path to the target retriever", required=True)
    parser.add_argument("--model_name", type=str, help="Name of the new model")
    parser.add_argument("--batch_size", type=int, help="Batch size for the model", default=32)
    parser.add_argument("--device", type=str, help="Device to run the model on", default='cpu')
    args = parser.parse_args()

    # Check if target_path directory exists, if not, create it
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)

    # Call the update_retriever function with the provided arguments
    update_retriever(args.source_path, args.target_path, args.model_name, args.batch_size, args.device)