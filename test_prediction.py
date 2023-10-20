from ANN_library import ANN
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()

model = ANN((16,16,16))

parameters = model.train(X, y, 0.05, 3000, True)

print(model.predict_value((0,0)))


