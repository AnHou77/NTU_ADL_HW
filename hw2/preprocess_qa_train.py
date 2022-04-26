import json

f = open('data/context.json',encoding='utf-8')
context = json.load(f)
f.close()

for mode in ['valid','train']:
    file_name = f'qa_{mode}.json'
    f = open(file_name,encoding='utf-8')
    train_data = json.load(f)
    f.close()

    new_train_data = []
    for data in train_data:
        temp = {}
        temp['id'] = data['id']
        temp['title'] = 'train'
        temp['context'] = context[data['relevant']]
        temp['question'] = data['question']
        answer = data['answer']
        answer['answer_start'] = [answer.pop('start')]
        answer['text'] = [answer['text']]
        temp['answers'] = answer
        new_train_data.append(temp)

    save_data = {}
    if mode == 'train':
        f = open(f'data/squad_valid.json',encoding='utf-8')
        valid = json.load(f)
        f.close()
        new_train_data = valid['data'] + new_train_data
    save_data['data'] = new_train_data

    with open(f'data/squad_{mode}.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)