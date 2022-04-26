import json

f = open(f'../hw1/cache/intent/intent2idx.json',encoding='utf-8')
intent2idx = json.load(f)
f.close()

for mode in ['eval','train','test']:
    f = open(f'../hw1/data/intent/{mode}.json',encoding='utf-8')
    train_data = json.load(f)
    f.close()

    new_train_data = []
    with open(f'it_{mode}.json', 'w', encoding='utf-8') as f:
        for data in train_data:
            temp = {}
            # temp['id'] = data['id']
            temp['text'] = data['text']
            if mode != 'test':
                temp['label'] = data['intent']
            else:
                temp['label'] = "definition"
            # new_train_data.append(temp)
        
            f.write(json.dumps(temp) + '\n')
    
    # with open(f'it_{mode}.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_train_data, f, ensure_ascii=False, indent=4)
    