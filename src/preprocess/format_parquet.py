import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, nargs="*", required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

fout = open(args.output_path, 'w')
for path in args.data_path:
    data = pd.read_parquet(path, engine='pyarrow')
    for i, row in data.iterrows():
        if row['text'].strip() != '':
            data = {'meta': {'filename': path.split('/')[-1], 'row': str(i)}, 'text': row['text']}
            data = json.dumps(data)
            fout.write(data + '\n')
fout.close()
