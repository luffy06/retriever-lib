import os
import lmdb
import pickle
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer

BERT_MODELS = ['bert-base-uncased', 'bert-large-uncased', 'robert-base', 'roberta-large']

def get_embeddings(model_name, embeddings):
    if model_name in BERT_MODELS:
        return embeddings[0, :].squeeze()
    else:
        return embeddings[-1, :].squeeze()

def update_db(path, new_path, model_name):
    # Open the LMDB environment
    env_old = lmdb.open(path, readonly=False)
    # Open the new LMDB environment
    env_new = lmdb.open(new_path, readonly=False)

    # Load the model based on the provided model_name
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Begin a write transaction
    txn_old = env_old.begin(write=False)
    txn_new = env_new.begin(write=True)

    # Iterate over all the keys in the LMDB
    cursor = txn_old.cursor()
    for key, value in cursor:
        # Deserialize the value (assuming it's stored as a pickled object)
        value = pickle.loads(value)

        # Generate a new embedding using the last hidden states of the model
        inputs = tokenizer(value['text'], return_tensors='pt')
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        new_embedding = get_embeddings(model_name.split('/')[-1], last_hidden_states[0].detach().numpy())

        # Update the value with the new embedding
        new_value = pickle.dumps({
            'text': value['text'],
            'embedding': new_embedding
        })

        # Store the updated value back into the LMDB
        txn_new.put(key, new_value)

    # Close the LMDB environment
    env_old.close()
    env_new.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, help="Path to the existing database")
    parser.add_argument("--new_db_path", type=str, help="Path to the updated database")
    parser.add_argument("--model_name", type=str, help="Name of the new model")
    args = parser.parse_args()

    # Check if new_db_path directory exists, if not, create it
    if not os.path.exists(args.new_db_path):
        os.makedirs(args.new_db_path)

    # Call the update_db function with the provided arguments
    update_db(args.db_path, args.new_db_path, args.model_name)