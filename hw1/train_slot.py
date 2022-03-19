from collections import Counter
import os
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import TagClsDataset
from utils import Vocab

from torch.utils.data import DataLoader
from model import TagClassifier

# import matplotlib.pyplot as plt

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    num_class = len(tag2idx)
    print(tag2idx)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, TagClsDataset] = {
        split: TagClsDataset(split_data, vocab, tag2idx, args.max_len, type='train')
        for split, split_data in data.items()
    }

    # i = 0
    # for d in data['train']:
    #     if i == 0:
    #         counter = Counter(d['tags'])
    #     else:
    #         counter += Counter(d['tags'])
    #     i += 1
    # print(counter)
    # it = counter.keys()
    # sum = 0
    # class_weight = []
    # for i in it:
    #     class_weight.append(counter[i])
    #     sum += counter[i]
    # print(class_weight)

    # (DONE) TODO: crecate DataLoader for train / dev datasets
    trainset_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, collate_fn=datasets['train'].collate_fn)
    evalset_loader = DataLoader(datasets['eval'], batch_size=args.batch_size, shuffle=False, collate_fn=datasets['eval'].collate_fn)
    
    # it = iter(trainset_loader)
    # data = it.next()
    # print(data['data'][0])
    # print(data['label'][0])
    # print(data['id'][0])
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # (DONE) TODO: init model and move model to target device(cpu / gpu)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model = TagClassifier(embeddings=embeddings, input_size=300, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=num_class)
    model.to(device)

    # (DONE) TODO: init optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0]).to(device),label_smoothing=0.1)

    best_acc = 0.0

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(1, args.num_epoch+1):
        
        model.train()
        
        train_loss = 0.0
        train_acc = 0.0
        
        for data in tqdm(trainset_loader,desc='Train'):
            optimizer.zero_grad()

            input = data['data'].to(device)
            label = data['label'].to(device)
            
            output = model(input)
            loss = criterion(output.permute(0,2,1),label)
            loss.backward()
            
            optimizer.step()

            acc = 0
            for i in range(len(output)):
                cnt = 0
                seq_len = data['seq_len'][i]
                pred = output[i,:seq_len,:].argmax(-1)
                gt = label[i][:seq_len]
                for s in range(seq_len):
                    if (pred[s] == gt[s]):
                        cnt += 1
                if cnt == seq_len:
                    acc += 1

            # Record Loss & Acc
            train_loss += loss.item()
            train_acc += (acc/len(output))
        
        train_loss = train_loss / len(trainset_loader)
        train_losses.append(train_loss)
        train_acc = train_acc / len(trainset_loader)
        train_accs.append(train_acc)
        print(f"[ Train | {epoch:03d}/{args.num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # Validation
        model.eval()

        valid_loss = 0.0
        valid_acc = 0.0
        
        for data in tqdm(evalset_loader, desc='Eval'):
            with torch.no_grad():

                input = data['data'].to(device)
                label = data['label'].to(device)
                
                output = model(input)
                loss = criterion(output.permute(0,2,1),label)

                acc = 0
                for i in range(len(output)):
                    cnt = 0
                    seq_len = data['seq_len'][i]
                    pred = output[i,:seq_len,:].argmax(-1)
                    gt = label[i][:seq_len]
                    for s in range(seq_len):
                        if (pred[s] == gt[s]):
                            cnt += 1
                    if cnt == seq_len:
                        acc += 1

                valid_loss += loss.item()
                valid_acc += (acc/len(output))
        
        valid_loss = valid_loss / len(evalset_loader)
        valid_losses.append(valid_loss)
        valid_acc = valid_acc / len(evalset_loader)
        valid_accs.append(valid_acc)

        print(f"[ Valid | {epoch:03d}/{args.num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('best acc: ',best_acc)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir,f'lstm_epoch_{epoch}.ckpt'))
            print('save model with acc:',best_acc)

    """  please install matplotlib first 
    plt.subplot(211)
    plt.plot(train_losses, 'r')
    plt.plot(valid_losses, 'b')
    plt.ylabel('Loss')
    plt.title('Training/Valid Loss & Accuracy')
    plt.legend(['train','valid'])

    plt.subplot(212)
    plt.plot(train_accs, 'r')
    plt.plot(valid_accs, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train','valid'])
    plt.savefig('acc_slot.png')
    """

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
