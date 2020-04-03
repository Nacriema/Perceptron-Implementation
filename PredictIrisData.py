'''
Training perceptron model trên dữ liệu Iris

Đầu tiên ta sử dụng pandas kết hợp với sklearn để load dữ liệu hoa ra thành dạng bảng Tabular, và in ra kiểm tra xem thứ
tự mà ta lấy ra đã đúng chưa

Tiếp theo, ta lấy 100 mẫu đầu theo thứ tự là 50 mẫu của setosa và 50 mẫu của versicolor, sau đó convert cái nhãn sang
thành 1 (versicolor) và -1 (setosa), ta gán nó cho vector y

Và cái mảng X là kiểu ndarray int chứa thông số 2 feature đó là chiều dài của sepal và petal rồi hiển thị thành scatter- plot

Khi vẽ đồ thị ra thì ta thấy sự phân phối của 2 loại hoa là phân biệt đối với 2 dữ liệu petal và sepal length. Đối với
2 chiều dữ liệu này, ta có thể thấy là một đường thẳng tuyến tính (linear decision boundary) là đủ để có thể phân biệt
giữa Setosa và Versicolor rồi.

Do đó, một cái linear classifier như là perceptron là đủ để phân biệt 2 loại hoa trong dũ liệu này hoàn hảo rồi.

Giờ là lúc train thuật toán perceptron trên tập Irís. Tương tự ta sẽ vẽ những điểm missclass trong mỗi epoch để check
xem thuật toán để xem là nó có thể phân tách 2 loại hoa thành 2 lớp được hay không.

Cái perceptron của mình chỉ cần 6 epoch để học được, và giờ có thể sử dụng để dự đoán tập train được dễ dàng. Giờ vẽ
thêm cái vùng mà nó đã hình thành và phân chia được

'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from PerceptronImplementation import Perceptron
from BoudaryVisual import plot_decision_regions

iris = load_iris()

# print(iris)

names = [iris['target_names'][i] for i in iris['target']]

print(names)

df = pd.DataFrame(data=np.c_[iris['data'], iris['target'], names], columns=iris['feature_names'] + ['target'] + ['target_names'])

# Dung cai tail de in ra 5 thang cuoi cung trong bang
print(df)

# Gio bieu dien duoi dang mau ve
'''
Ta lấy 100 dũ liệu đầu tiên và lấy 2 cái thuộc tính là (sepal_length) và thuộc tính thứ 3 (petal length) để biểu diễn nó
trong đồ thị điểm 2 chiều
'''

# Chon setosa va versicolor
y = df.iloc[0: 100, 5].values
y = np.where(y == 'setosa', -1, 1)

# Lay 2 thong so sepal length va petal length
X = df.iloc[0: 100, [0, 2]].values
# Chuyen cac thanh phan ben trong X sang kieu so de no co the sort lai va ve ra dung duoc
'''Source: https://stackoverflow.com/questions/51341717/xlabel-and-ylabel-values-are-not-sorted-in-matplotlib-scatterplot/51345033'''
X = X.astype(float)

# plot data su dung subplot
plt.subplot(2, 2, 1)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc=0)
print(X)


ppn = Perceptron(eta=0.5, n_iter=10)
ppn.fit(X, y)
plt.subplot(2, 2, 2)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of updates')
print(ppn.errors_)
#plt.show()

plt.subplot(2, 2, 3)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

plt.legend(loc='upper left')
plt.show()
