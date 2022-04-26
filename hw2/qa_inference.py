import json
import pandas as pd
import sys


if __name__ == '__main__':
    output_file = sys.argv[1]
    f = open('output/predict_predictions.json',encoding='utf-8')
    context = json.load(f)
    f.close()

    keys = [] 
    values = []
    for k,v in context.items():
        keys.append(k)
        values.append(v)

    df = pd.DataFrame({'id':keys,'answer':values})
    df.to_csv(output_file,index=False)