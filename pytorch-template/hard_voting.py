import numpy as np
import pandas as pd
import collections

answers = []
one_hot = []

for i in range(1,18):
    answers.append(pd.read_csv(str(i)+'.csv'))

for i in range(17):
    one_hot.append(pd.get_dummies(answers[i]['ans']))


best = one_hot[0].to_numpy()


zeros = np.zeros([12600,18])

for i in range(17):
    zeros += one_hot[i].to_numpy()


ans = []
for i in range(len(zeros)):
    if collections.Counter(zeros[i])[np.max(zeros[i])] == 1:
        ans.append(np.argmax(zeros[i]))
        continue
    elif collections.Counter(zeros[i])[np.max(zeros[i])] >= 2:
        ans.append(np.argmax(best[i]))


answers[0]['ans'] = pd.Series(ans)

answers[0].to_csv('hard_voting.csv')