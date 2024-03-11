import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import plot_tree

from clearml import Task


task = Task.init(project_name='test_project', task_name='XGBoost simple example', output_uri=True)

fashion_mnist_test = pd.read_csv("data/fashion_test.csv")
fashion_mnist_train = pd.read_csv("data/fashion_train.csv")

X_train = np.array(fashion_mnist_train.iloc[:,1:])
y_train = np.array(fashion_mnist_train.iloc[:,0])
X_test = np.array(fashion_mnist_test.iloc[:,1:])
y_test = np.array(fashion_mnist_test.iloc[:,0])


plt.imshow(X_train[1].reshape((28, 28)))
plt.title("Sample Image")
plt.show()


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 3,  # the maximum depth of each tree
    "eta": 0.3,  # the training step for each iteration
    "gamma": 0,
    "max_delta_step": 1,
    "subsample": 1,
    "sampling_method": "uniform",
    "seed": 1337
}
task.connect(params)


bst = xgb.train(
    params,
    dtrain,
    num_boost_round=25,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0,
)

bst.save_model("best_model.ubj")


y_pred = bst.predict(dtest)
predictions = [round(value) for value in y_pred]


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


plot_tree(bst)
plt.title("Decision Tree")
plt.show()

