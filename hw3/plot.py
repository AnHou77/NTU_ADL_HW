from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import sys

source_path = sys.argv[1]
target_path = sys.argv[2]

source = pd.read_csv(source_path)

rouge_1 = source['rouge-1'].tolist()
rouge_2 = source['rouge-2'].tolist()
rouge_l = source['rouge-l'].tolist()

epochs = [epoch for epoch in range(len(rouge_1))]

plt.plot(epochs,rouge_1,marker='o',label='rouge-1')
plt.plot(epochs,rouge_2,marker='o',label='rouge-2')
plt.plot(epochs,rouge_l,marker='o',label='rouge-L')
plt.title('Learning Curve')
plt.ylabel('Rouge score')
plt.xlabel('Number of epoch')
plt.xticks(epochs,epochs)
plt.legend(loc='center right')
plt.tight_layout()
plt.savefig(target_path)