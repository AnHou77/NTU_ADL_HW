import json
import sys


if __name__ == '__main__':
    context_file = sys.argv[1]
    test_file = sys.argv[2]

    f = open(context_file,encoding='utf-8')
    context = json.load(f)
    f.close()

    f = open(f'qa_test.json',encoding='utf-8')
    test_data = json.load(f)
    f.close()

    new_train_data = []
    for data in test_data:
        temp = {}
        temp['id'] = data['id']
        temp['title'] = 'train'
        temp['context'] = context[data['relevant']]
        temp['question'] = data['question']
        answer = {}
        answer['answer_start'] = [0]
        answer['text'] = ['']
        temp['answers'] = answer
        new_train_data.append(temp)

    save_data = {}
    save_data['data'] = new_train_data

    with open(f'test.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)