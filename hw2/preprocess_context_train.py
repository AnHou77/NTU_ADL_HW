import json
import pandas as pd

f = open('data/context.json',encoding='utf-8')
context = json.load(f)
f.close()

for mode in ['valid','train']:
    f = open(f'data/{mode}.json',encoding='utf-8')
    train_data = json.load(f)
    f.close()

    new_train_data = []
    for data in train_data:
        temp = {}
        temp['video-id'] = data['id']
        temp['fold-ind'] = 'none'
        temp['startphrase'] = 'none'
        temp['sent1'] = data['question']
        temp['sent2'] = ''
        temp['gold-source'] = 'none'
        paragraphs = data['paragraphs']
        temp['ending0'] = context[paragraphs[0]]
        temp['ending1'] = context[paragraphs[1]]
        temp['ending2'] = context[paragraphs[2]]
        temp['ending3'] = context[paragraphs[3]]
        temp['label'] = paragraphs.index(data['relevant'])
        new_train_data.append(temp)

    if mode == 'valid':
        with open(f'data/swag_{mode}.json', 'w', encoding='utf-8') as f:
            json.dump(new_train_data, f, ensure_ascii=False, indent=4)

    if mode == 'train':
        f = open(f'data/swag_valid.json',encoding='utf-8')
        train = json.load(f)
        f.close()
        new_train_data = train + new_train_data
    
    df = pd.DataFrame(new_train_data)
    df.to_csv(f'data/swag_{mode}.csv')