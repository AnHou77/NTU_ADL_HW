python download.py

mkdir -p 'cache/intent'
mkdir -p 'cache/slot'

# Intent Classification
wget -O './intent_best.ckpt' 'https://www.dropbox.com/s/bd6iwev3wtxvlkk/intent_best.ckpt?dl=1'
wget -O 'cache/intent/embeddings.pt' 'https://www.dropbox.com/s/jvxc1jeg7jvn6js/intent_embeddings.pt?dl=1'
wget -O 'cache/intent/intent2idx.json' 'https://www.dropbox.com/s/u9tc4s0mi1ollgu/intent2idx.json?dl=1'
wget -O 'cache/intent/vocab.pkl' 'https://www.dropbox.com/s/zh4boxr5aebsxv8/intent_vocab.pkl?dl=1'

# Slot Tagging
wget -O './slot_best.ckpt' 'https://www.dropbox.com/s/r8jyy3gqj1shva9/slot_best.ckpt?dl=1'
wget -O 'cache/slot/embeddings.pt' 'https://www.dropbox.com/s/lmb4j2enwpgaqp6/slot_embeddings.pt?dl=1'
wget -O 'cache/slot/tag2idx.json' 'https://www.dropbox.com/s/1p6iz6c9it5rmbk/tag2idx.json?dl=1'
wget -O 'cache/slot/vocab.pkl' 'https://www.dropbox.com/s/fygitxbpmhelpno/slot_vocab.pkl?dl=1'