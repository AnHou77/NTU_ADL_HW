# Homework 1 ADL NTU 110 Spring

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
# training script
python train_intent.py

# inference script
bash intent_cls.sh {test_file} {pred_csv}
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
        dropout=0.2,
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

## Slot tagging
```shell
# training script
python train_slot.py

# inference script
bash slot_tag.sh {test_file} {pred_csv}
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
- ![image](https://github.com/AnHou77/NTU_ADL_HW/tree/master/hw1/acc_slot.png)