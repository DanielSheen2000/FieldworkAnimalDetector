import torch
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from local import *
import seaborn as sns

model = torch.load('tmpmodel.pt')
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
test_dataset = testingDataset(X_test, y_test)
test_loader = data.DataLoader(test_dataset, batch_size=test_dataset.__len__())
with torch.no_grad():
    for batch in test_loader:
        tmp_prediction = []
        input, label = batch
        # Obtaining our mfcc, delta and delta_delta from X and converting to tensor
        for j in range(len(input)):
            mfcc = input[j][0].unsqueeze(0).unsqueeze(0)
            delta = input[j][1].unsqueeze(0).unsqueeze(0)
            delta_delta = input[j][2].unsqueeze(0).unsqueeze(0)
            prediction = model(mfcc, delta, delta_delta)
            tmp_prediction.append(prediction)
        tmp_prediction = torch.tensor(tmp_prediction, requires_grad=True)
        for i in range(len(tmp_prediction)):
            if tmp_prediction[i] >= 0.5 and label[i] == 1:
                true_positive += 1
            elif tmp_prediction[i] < 0.5 and label[i] == 0:
                true_negative += 1
            elif tmp_prediction[i] >= 0.5 and label[i] == 0:
                false_positive += 1
            elif tmp_prediction[i] < 0.5 and label[i] == 1:
                false_negative += 1

cm = ([true_positive,false_negative],[false_positive,true_negative])
x_axis = y_axis = ['Lions','Hyenas']
sns.heatmap(cm,annot=True,fmt='d',xticklabels=x_axis,yticklabels=y_axis)
plt.title('Confusion Matrix')
plt.savefig('confusion matrix')
plt.show()