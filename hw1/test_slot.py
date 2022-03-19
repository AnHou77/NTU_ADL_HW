import json
from operator import index
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import TagClsDataset
from model import TagClassifier
from utils import Vocab

from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    num_class = len(tag2idx)

    data = json.loads(args.test_file.read_text())
    dataset = TagClsDataset(data, vocab, tag2idx, args.max_len, type='test')
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model = TagClassifier(embeddings=embeddings, input_size=300, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=num_class)
    
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(device)

    model.eval()
    
    id = []
    preds = []
    for data in tqdm(dataloader,desc='inference'):
        with torch.no_grad():
            input = data['data'].to(device)
            output = model(input)
            seq_len = data['seq_len'][0]
            pred = output[0,:seq_len,:].argmax(-1)
            sequence = ''
            for p in pred:
                sequence += dataset.idx2label(p.item())
                sequence += ' '
            sequence = sequence[:-1]
            id.append(data['id'][0])
            preds.append(sequence)
    
    df = pd.DataFrame({'id': id, 'tags':preds})
    df.to_csv(args.pred_file,index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=36)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--num_class", type=int, default=150)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
