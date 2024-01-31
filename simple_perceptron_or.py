def perceptron(input1, input2):
    weights = [1, 1]  # Weights for inputs
    bias = 0  # Bias term

    weighted_sum = input1 * weights[0] + input2 * weights[1] + bias

    return weighted_sum > 0;

print(perceptron(0, 0))  # Output: 0
print(perceptron(0, 1))  # Output: 1
print(perceptron(1, 0))  # Output: 1
print(perceptron(1, 1))  # Output: 1
