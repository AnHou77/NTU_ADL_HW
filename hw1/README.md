# Homework 1 ADL NTU 109 Spring

## Environment
```shell
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
## Hyper parameters in training process ##
# data
parser.add_argument("--max_len", type=int, default=32)

# model
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--bidirectional", type=bool, default=True)
parser.add_argument("--num_class", type=int, default=150)

# optimizer
parser.add_argument("--lr", type=float, default=1e-3)

# data loader
parser.add_argument("--batch_size", type=int, default=256)

# training
parser.add_argument(
    "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
)
parser.add_argument("--num_epoch", type=int, default=100)
##########################################

# training script
python train_intent.py
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
        dropout=0.2,
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


### improved
- performance:
```
Kaggle score:   0.9284
```
- data:
    ```
    embedding max_len (seq_len): 32
    batch size: 256
    ```
- model: (same as baseline)
- optimizer: (same as baseline)


## Slot tagging
