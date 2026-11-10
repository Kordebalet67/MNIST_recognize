from MNIST import MNIST_Neural_Network as net


def main():
    new_net = net(784, 392, 10)
    new_net.load_weights('weights(784,392,10).joblib')

    with open("mnist_test.csv", 'r') as file:
        test_data_list = file.readlines()
    new_net.score_info(test_data_list)

    answers = []
    confidences = []

    for i in range(0, 10):
        answer, confidence = new_net.input_from_picture(f'pic_{i}.jpg')
        answers.append(answer)
        confidences.append(confidence[0])

    for i in range(0, 10):
        print(f'{answers[i]}: {confidences[i]:.4f} %')

if __name__ == '__main__':
    main()