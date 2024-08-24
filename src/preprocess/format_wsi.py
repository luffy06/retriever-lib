import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Format WSI')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    id_info = []
    embeddings = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(args.input_dir, filename)
            # Read the pkl file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            for key, value in data.items():
                key = '.'.join(key.split('.')[:-1])
                wsi_id = key.split('_')[0]
                patch_id = (int(key.split('_')[2]), int(key.split('_')[3]))
                id_info.append((wsi_id, patch_id))
                embeddings.append(value)
    embeddings = np.array(embeddings)
    df = pd.DataFrame({'embedding': embeddings.tolist()})
    emb_dir = os.path.join(args.output_dir, 'emb')
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    metadata = {'num_emb': embeddings.shape[0],
                'emb_dim': embeddings.shape[1]}
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as fout:
        json.dump(metadata, fout)
    emb_path = os.path.join(emb_dir, 'wsi-feature-0.json.gz')
    df.to_json(emb_path, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})

    value_dir = os.path.join(args.output_dir, 'value')
    if not os.path.exists(value_dir):
        os.makedirs(value_dir)
    value_path = os.path.join(value_dir, 'value.pkl')
    with open(value_path, 'wb') as file:
        pickle.dump(id_info, file)

if __name__ == '__main__':
    main()