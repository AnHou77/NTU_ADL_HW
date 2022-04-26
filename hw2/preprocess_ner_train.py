import json

for mode in ['eval','train','test']:
    f = open(f'../hw1/data/slot/{mode}.json',encoding='utf-8')
    train_data = json.load(f)
    f.close()

    new_train_data = []
    with open(f'slot_{mode}.json', 'w', encoding='utf-8') as f:
        for data in train_data:
            temp = {}
            temp['id'] = data['id']
            temp['tokens'] = data['tokens']
            z = []
            if mode != 'test':
                tags = data['tags']
                ner = []
            for i in range(len(temp['tokens'])):
                z.append('O')
                if mode != 'test':
                    ner.append(tags[i])
            temp['pos_tags'] = z
            temp['chunk_tags'] = z
            if mode != 'test':
                temp['ner_tags'] = ner
            else:
                temp['ner_tags'] = z
            # new_train_data.append(temp)
            f.write(json.dumps(temp) + '\n')

    f.close()
        # json.dump(new_train_data, f, ensure_ascii=False, indent=4)