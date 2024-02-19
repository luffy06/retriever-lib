# retriever-lib

This library integrates existing retrievers, and provides a series of unified interfaces to build the retrievers over different corpus texts.

The building workflow of this library includes the following steps,
1. Prepare the corpus in json-formatted files;
2. Chunk the corpus;
3. Encode the chunks into embeddings;
4. Build the retrieval index based on the embeddings;
5. Build the retrieval database according to the task.

## Corpus

The corpus for retrievers can be any text, such as wikipedia data.

We provide the script `donwload_corpus.sh` to download two kinds of corpus, i.e., [wikitext-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) and [wikipedia-split](https://github.com/facebookresearch/DPR), you can run the script to download them and format them into json files.

```bash
bash scripts/download_corpus.sh
```

In the script, you should specify the root path of current project `PROJECT_DIR`.
Then, all original downloaded files will be stored in the directory `PROJECT_DIR/corpus/original`.
All formatted json files will be in the directory `PROJECT_DIR/corpus/formatted`.

## Chunking and Encoding

The corpus is always comprised of many documents, and it is difficult to directly encode those documents into embeddings due to long length. Thus, chunking as a common technique is introduced to split the original docments into shorter chunks for better semantic representation.

We implement various chunking methods,
1. Chunk-by-Sentence. This method chunks documents based on sentence end words, such as '.' and '!'.

After chunking, we use language models to encode chunks into embeddings. We implement the encoding codes based on the [Sentence Transformer](https://github.com/UKPLab/sentence-transformers).

To run the chunking and encoding, you can run the following codes,
```bash
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $DATA_DIR \
    --model_path $ENCODER_PATH \
    --output_dir $OUTPUT_DIR \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 1000000 \
    --batch_size 256 \
```
, where `$DATA_DIR`, `$ENCODER_PATH`, `$OUTPUT_DIR` refer to the data directory path stroing the formatted json files, the path of encoder model, and the output directory path.

## Building Retrieval Indexes

The retrieval index is used to accelerate the searching process among billion-scale retrieval database. There are three types of retrieval indexes, the sparse, the dense, and the model-based.

### Dense Retrieval Indexes

We encapuslate various dense retrieval indexes, 
1. [faiss](https://github.com/facebookresearch/faiss), to support billion-scale retrievals, we choose `IVF*_HNSW*,PQ*` as the dense index. The building process involves training the base index, building sub indexes, and merging all sub indexes.

## Building Retrieval Database

The retrieval database is a key-value store, where keys are embedding ids, and values are task-specific items. The key point in the retrieval database is how to design values.

Different tasks need different values,
1. Default. The default values include the corresponding text and the embedding itself.
2. Language modeling. The values include the text and the next text (few tokens).

To build the index and database, you can run the following codes,
```bash
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $DATA_DIR \
    --model_path $ENCODER_PATH \
    --output_dir $OUTPUT_DIR \
    --device_id 0 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
```
You can refer to the [guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) to choose the index type.

Overall, to pipeline the whole process, you can directly run the follow script for different corpus,
```bash
bash script/build_retriever_wikitext_103.sh
```

## Run the Built Retriever

We have provide a class to load the built retriever and search on the retrieval database. 

To create an instance of the class, you need to at least pass the directory of retriever, the nprobe for faiss search, the top-k neighbors.
```python
retriever = Retriever(
    retriever_dir=args.retriever_dir, 
    nprobe=args.nprobe, 
    topk=args.topk, 
)
```

Then, you can call the `search()` function to retrieve the top-k nearest neighbors,
```python
neighbors = retriever.search(query_embeddings)
```

The results are in the form of dict, where keys are query ids, and values are the nearest neighbors.