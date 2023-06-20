from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

class Evaluation:
    def __init__(self, y_test, y_pred):
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred)
        self.recall = recall_score(y_test, y_pred)
        try:
            self.auc = roc_auc_score(y_test, y_pred)
        except:
            self.auc = 0
        self.f1 = f1_score(y_test, y_pred)
        self.cm = confusion_matrix(y_test, y_pred)

    def print(self):
        print("Accuracy : ", round(self.accuracy, 2))
        print("Precision : ", round(self.precision, 2))
        print("Recall : ", round(self.recall, 2))
        print("AUC : ", round(self.auc, 2))
        print("F1 score : ", round(self.f1, 2))
        print("Confusion Matrix : \n", self.cm)