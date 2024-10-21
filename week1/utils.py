import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch

EPS = 0.2
def show_boundary_sklearn(model, data, x1_name='sepal_length', x2_name='petal_length', y_name='species'):
  x1, x2 = np.meshgrid(np.arange(min(data[x1_name]) - EPS, max(data[x1_name]) + EPS, 0.001),
                       np.arange(min(data[x2_name]) - EPS, max(data[x2_name]) + EPS, 0.001))
  grid = np.stack([x1.flatten(), x2.flatten()]).T
  Z = model.predict(grid)
  color_list = ['red' if elem == 1 else 'blue' for elem in data[y_name]]   # 1 == versicolor
  Z = Z.reshape(*x1.shape)
  plt.contourf(x1, x2, (Z < 0.5), cmap="RdBu", alpha=0.5)

def train_model_for_wheat(wheat_data):
  logreg = LogisticRegression()
  logreg.fit(X=wheat_data[['kernel_groove_length', 'asymmetry']], y=wheat_data['type'])
  color_list = ['red' if elem == 1 else 'blue' for elem in wheat_data['type']]
  plt.scatter(wheat_data['kernel_groove_length'], wheat_data['asymmetry'], c=color_list)
  show_boundary_sklearn(logreg, wheat_data, x1_name='kernel_groove_length',
                                    x2_name='asymmetry',
                                    y_name='type')

  plt.xlabel("Kernel Groove Length")
  plt.ylabel("Assymetry Coefficient")
  plt.title("Red = Kama, blue = Canadian")
  plt.show()
  accuracy = logreg.score(X=wheat_data[['kernel_groove_length', 'asymmetry']], y=wheat_data['type'])
  print(f"Accuracy: {accuracy*100:.1f}%")

def show_boundary_pytorch(model, X, y):
  x1, x2 = torch.meshgrid(torch.arange(min(X[:,0]) - EPS, max(X[:,0]) + EPS, 0.001),
                          torch.arange(min(X[:,1]) - EPS, max(X[:,1]) + EPS, 0.001))
  grid = torch.stack([x1.flatten(), x2.flatten()]).T
  Z = model(grid)
  color_list = ['red' if elem == 1 else 'blue' for elem in y]   # 1 == versicolor
  Z = Z.reshape(*x1.shape)
  plt.contourf(x1, x2, (Z < 0.5), cmap="RdBu", alpha=0.5)

def plot_progress(model, X_train, X_test, y_train, y_test, epochs):
  color_list = ['red' if elem == 1 else 'blue' for elem in y_train]
  plt.scatter(X_train[:,0], X_train[:,1], c=color_list)
  show_boundary_pytorch(model, X_train, y_train)
  plt.xlabel("Sepal Length")
  plt.ylabel("Petal Length")
  plt.title("Training Data Decision Boundary After {} Epochs".format(epochs))
  plt.show()

  color_list = ['red' if elem == 1 else 'blue' for elem in y_test]
  plt.scatter(X_test[:,0], X_test[:,1], c=color_list)
  show_boundary_pytorch(model, X_test, y_test)
  plt.xlabel("Sepal Length")
  plt.ylabel("Petal Length")
  plt.title("Testing Data Decision Boundary After {} Epochs".format(epochs))
  plt.show()

def numpy_to_tensor(*args):
  return tuple(torch.from_numpy(arg).float() for arg in args)

def binary_accuracy(y_pred, y_true):
  y_pred = (y_pred >= 0.5).float()
  correct = (y_pred == y_true).float()
  accuracy = correct.sum() / len(correct)
  return accuracy.item()