import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

fin = open(args.data_path, 'r')
fout = open(args.output_path, 'w')

row = -1
while True:
  data = fin.readline().strip()
  if data == '':
    break
  row = row + 1
  if row == 0:
    continue
  data = data.split('\t')
  data = {'meta': {'filename': args.data_path.split('/')[-1], 'row': str(row), 'title': data[2]}, 'text': data[1]}
  data = json.dumps(data)
  fout.write(data + '\n')

fin.close()
fout.close()
