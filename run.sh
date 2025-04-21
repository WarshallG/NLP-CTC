# pip install tqdm numpy torch gensim jieba scikit-learn matplotlib transformers==4.46.3 Levenshtein

python main.py \
    --train_file data/train.jsonl \
    --test_file data/test.jsonl \
    --method rule \
    # --analyze

python main.py \
    --train_file data/train.jsonl \
    --test_file data/test.jsonl \
    --method statistical \
    --statistical_method ngram \
    # --analyze

python main.py \
    --train_file data/train.jsonl \
    --test_file data/test.jsonl \
    --method statistical \
    --statistical_method ml \
    # --analyze

python main.py \
    --train_file data/train.jsonl \
    --test_file data/test.jsonl \
    --method ensemble \
    --statistical_method ngram \
    # --analyze

python main.py \
    --train_file data/train.jsonl \
    --test_file data/test.jsonl \
    --method ensemble \
    --statistical_method ml \
    # --analyze