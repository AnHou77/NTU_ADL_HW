import json
import pandas as pd
import sys


if __name__ == '__main__':
    context_file = sys.argv[1]
    test_file = sys.argv[2]

    f = open(context_file,encoding='utf-8')
    context = json.load(f)
    f.close()

    f = open(test_file,encoding='utf-8')
    test_data = json.load(f)
    f.close()

    new_test_data = []
    for data in test_data:
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
        temp['label'] = 0
        new_test_data.append(temp)
    
    df = pd.DataFrame(new_test_data)
    df.to_csv(f'cs_test.csv')