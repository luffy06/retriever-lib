import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd

def read_gene_name(gene_name_path):
    with open(gene_name_path, 'r') as f:
        gene_names = f.readlines()
    return gene_names

def main():
    parser = argparse.ArgumentParser(description='Format WSI')
    parser.add_argument('--input_dir', type=str, required=True, help='Gene name path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    img_path = os.path.join(args.input_dir, 'image_features.npy')
    gene_value_path = os.path.join(args.input_dir, 'gene_values.npy')

    img_features = np.load(img_path)
    gene_values = np.load(gene_value_path)

    embeddings = np.array(img_features)
    df = pd.DataFrame({'embedding': embeddings.tolist()})
    emb_dir = os.path.join(args.output_dir, 'emb')
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    metadata = {'num_emb': embeddings.shape[0],
                'emb_dim': embeddings.shape[1]}
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as fout:
        json.dump(metadata, fout)
    emb_path = os.path.join(emb_dir, 'img-feature-0.json.gz')
    df.to_json(emb_path, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})

    value_dir = os.path.join(args.output_dir, 'value')
    if not os.path.exists(value_dir):
        os.makedirs(value_dir)
    value_path = os.path.join(value_dir, 'value.pkl')
    with open(value_path, 'wb') as file:
        pickle.dump(gene_values, file)

if __name__ == '__main__':
    main()