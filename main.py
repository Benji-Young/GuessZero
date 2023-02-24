import math
import matplotlib.pyplot as plt

target = 0.0
net_in = 1.0
net_weight = 0.6
net_bias = 0.9
lr = 0.15

error_list = []


# functions used
def sigmoid_function(x):
    return 1.0 / (1.0 + math.exp(-x))


def cost_function(x, a):
    return math.pow(a - x, 2.0)


# derivatives
def cost_output_derivative(x, a):
    return 2.0 * (x - a)


# x is the weighted input
def output_activation_derivative(x):
    return sigmoid_function(x) * (1.0 - sigmoid_function(x))


# Train the network
epoch = []
for i in range(200):
    epoch.append(i)
    output = sigmoid_function(net_in * net_weight + net_bias)
    error_list.append(cost_function(output, target))
    net_weight -= (cost_output_derivative(output, target) * output_activation_derivative(
        net_in * net_weight) * (net_in * net_weight)) * lr
    net_bias -= (output_activation_derivative(net_in * net_weight)) * lr
    print(output)

print(net_weight, net_bias)

plt.plot(epoch, error_list)
plt.xlabel("epoch")
plt.ylabel("error")
plt.title("Net Performance")
plt.show()
