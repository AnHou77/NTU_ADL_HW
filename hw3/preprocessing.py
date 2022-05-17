import jsonlines
import json
import sys
with open(sys.argv[1], "r", encoding="utf8") as f:
    with open(sys.argv[2],'w',encoding='utf8') as fout:
        for item in jsonlines.Reader(f):
            temp = {}
            temp['id'] = item['id']
            temp['title'] = item['title']
            temp['text'] = item['maintext']
            fout.write(json.dumps(temp) + '\n')
f.close()
fout.close()