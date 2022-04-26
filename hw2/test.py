import matplotlib.pyplot as plt
import pandas as pd

# loss = [0.9058,0.7842,0.478,0.4958,0.4802]
# EM = [81.6883,87.1054,90.4952,92.7883,94.6162]

# epoch = [0.4,0.8,1.2,1.6,2.0]
# # plt.plot(epoch,loss)
# # plt.title('QA model loss curve')
# # plt.ylabel('Loss')
# # plt.xlabel('Epoch')
# # plt.savefig('loss_curve.png')

# plt.plot(epoch,EM)
# plt.title('QA model EM curve')
# plt.ylabel('EM')
# plt.xlabel('Epoch')
# plt.savefig('em_curve.png')


# with open('tmp/slot/predictions.txt') as f:
#     line = f.readlines()
# f.close()

# ids = []
# tags = []
# for i in range(len(line)):
#     ids.append(f'test-{i}')
#     tag = line[i].rstrip('\n')
#     tags.append(tag)

# df = pd.DataFrame({'id':ids,'tags':tags})
# df.to_csv('slot_submission.csv',index=False)

# with open('tmp/intent/predict_results_None.txt') as f:
#     line = f.readlines()
# f.close()


# ids = []
# intent = []
# for i in range(len(line)):
#     if i != 0:
#         ids.append(f'test-{i-1}')
#         it = line[i].split('	')[1].rstrip('\n')
#         intent.append(it)

# df = pd.DataFrame({'id':ids,'intent':intent})
# df.to_csv('intent_submission.csv',index=False)