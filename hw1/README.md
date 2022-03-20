# Homework 1 ADL NTU 110 Spring

## Environment
```shell
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
## for training
bash preprocess.sh

## for inference
bash download.sh
```

## Intent detection
```shell
# training script
python train_intent.py

# inference script
bash intent_cls.sh {test_file} {pred_csv}
```
### Parameters (train_intent.py)
```shell
parser.add_argument("--data_dir", type=Path, default="./data/intent/")
parser.add_argument("--cache_dir", type=Path, default="./cache/intent/")
parser.add_argument("--ckpt_dir", type=Path, default="./ckpt/intent/")

# data
parser.add_argument("--max_len", type=int, default=32)

# model
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--bidirectional", type=bool, default=True)
parser.add_argument("--num_class", type=int, default=150)

# optimizer
parser.add_argument("--lr", type=float, default=1e-3)

# data loader
parser.add_argument("--batch_size", type=int, default=256)

# training
parser.add_argument("--device", type=torch.device, default="cuda:0")
parser.add_argument("--num_epoch", type=int, default=10)
```

### Parameters (test_intent.py)
```shell
parser.add_argument("--test_file", type=Path, required=True)
parser.add_argument("--cache_dir", type=Path, default="./cache/intent/")
parser.add_argument("--ckpt_dir", type=Path, required=True)
parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

# data
parser.add_argument("--max_len", type=int, default=32)

# model
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--bidirectional", type=bool, default=True)
parser.add_argument("--num_class", type=int, default=150)

# data loader
parser.add_argument("--batch_size", type=int, default=1)

# Testing
parser.add_argument("--device", type=torch.device, default="cuda:0")
```

### baseline
- performance:
    ```
    Kaggle score:   0.9133
    ```
- data:
    ```
    embedding max_len (seq_len): 128
    batch size: 256
    ```
- model:
    ```
    LSTM(
        input_size=300,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        batch_first=True
    )
    
    fc = nn.Sequential(
        nn.Linear(seq_len*hidden_size*2,1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024,num_class),
        nn.Sigmoid()
    )
    ```
- optimizer:
    ```
    Adam,
    learning rate: 1e-3
    ```

### improved (Final)
- performance:
```
Kaggle score:   0.9377
```
- data: 
    ```
    embedding max_len (seq_len): 32
    batch size: 256
    ```
- model: 
    ```
    LSTM(
        input_size=300,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        batch_first=True
    )
    
    fc = nn.Sequential(
        nn.Linear(seq_len*hidden_size,1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024,num_class),
    )
    ```
- optimizer:
    ```
    Adam,
    learning rate: 1e-3

- ![image](https://github.com/AnHou77/NTU_ADL_HW/blob/master/hw1/acc.png)

### Plot figures
- Uncommand (line 19), (line 133-148) in train_intent.py & pip install matplotlib

## Slot tagging
```shell
# training script
python train_slot.py

# inference script
bash slot_tag.sh {test_file} {pred_csv}
```
### Parameters (train_slot.py)
```shell
parser.add_argument("--data_dir", type=Path, default="./data/slot/")
parser.add_argument("--cache_dir", type=Path, default="./cache/slot/")
parser.add_argument("--ckpt_dir", type=Path, default="./ckpt/slot/")

# data
parser.add_argument("--max_len", type=int, default=36)

# model
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--bidirectional", type=bool, default=True)

# optimizer
parser.add_argument("--lr", type=float, default=1e-3)

# data loader
parser.add_argument("--batch_size", type=int, default=256)

# training
parser.add_argument("--device", type=torch.device, default="cuda:0")
parser.add_argument("--num_epoch", type=int, default=50)
```

### Parameters (test_slot.py)
```shell
parser.add_argument("--test_file", type=Path, required=True)
parser.add_argument("--cache_dir", type=Path, default="./cache/slot/")
parser.add_argument("--ckpt_dir", type=Path, required=True)
parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

# data
parser.add_argument("--max_len", type=int, default=36)

# model
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--bidirectional", type=bool, default=True)

# data loader
parser.add_argument("--batch_size", type=int, default=1)

# Testing
parser.add_argument("--device", type=torch.device, default="cuda:0")
```

### baseline
- performance:
    ```
    Kaggle score:   0.7142
    ```
- data:
    ```
    embedding max_len (seq_len): 36
    batch size: 256
    ```
- model:
    ```
    LSTM(
        input_size=300,
        hidden_size=1024,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        batch_first=True
    )
    
    fc = nn.Linear(hidden_size * 2,num_class)
    ```
- optimizer:
    ```
    Adam,
    learning rate: 1e-3
    ```
- loss:
    ```
    CrossEntropyLoss,
    label smooth = 0.1
    ```
### improved (Final)
- performance:
    ```
    Kaggle score:   0.7705
    ```
- data:
    ```
    embedding max_len (seq_len): 36
    batch size: 256
    ```
- model:
    ```
    LSTM(
        input_size=300,
        hidden_size=1024,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        batch_first=True
    )
    
    fc = nn.Linear(hidden_size * 2,num_class)
    ```
- optimizer:
    ```
    Adam,
    learning rate: 1e-3
    ```
- loss:
    ```
    CrossEntropyLoss,
    label smooth: 0.1,
    weight balance (for each class in tag2idx): [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    ```
- ![image](https://github.com/AnHou77/NTU_ADL_HW/blob/master/hw1/acc_slot.png)

### Plot figures
- Uncommand (line 19), (line 174-189) in train_slot.py & pip install matplotlib