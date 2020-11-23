import numpy as np
import math
import time
'''
def basic_sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s
x =np.array( [1 ,2 ,3 ])

print(basic_sigmoid(x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2],1)
    return v
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image2vector(image) = "+ str(image2vector(image)))


x = np.array([2, 6, 4])
x1 = np.linalg.norm(x, axis = 1, keepdims = True)
x_normalized = x / x1
print(x_normalized)

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    print(np.shape(x_norm))
    return x

x = np.array([[0, 3, 4],[1, 6, 4]])
print("normalizeRows = "+ str(normalizeRows(x)))
'''
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims=True)
    s = x_exp / x_sum
    return s

x = np.array([[9, 2, 5, 0, 0],[7, 5, 0, 0, 0]])

print("softmax(x) = " + str(softmax(x)))

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print("dot = " + str(dot) + 'time= ' + str(1000*(toc-tic)))