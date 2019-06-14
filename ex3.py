import numpy as np
import matplotlib.pyplot as plt

NUM_OF_CLASSES = 10

# Neural Network hyper params
LEARNING_RATE = 0.05
TRAIN_VALIDATION_RATIO = 0.85
NUM_OF_EPOCHS = 35
MINI_BATCH_SIZE = 32
HIDDEN_LAYER_SIZE = 100


def show_accuracy_graph(accuracies, title):
    plt.plot([i for i in range(len(accuracies))], accuracies, '--o')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch Number")
    plt.title(title)


def write_predictions(output_path, test_predictions):
    with open(output_path, "w") as output_file:
        output_file.write("\n".join(map(str, test_predictions)))


def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)


def reLU(x):
    return np.maximum(x, 0)


def reLU_gradient(x):
    return np.maximum(np.sign(x), 0)


class NeuralNetwork(object):
    """
    Neural Network with 1 hidden layer
    """
    def __init__(self, num_of_classes, train_set):
        self.num_of_classes = num_of_classes
        self.train_set = train_set

        # Weights of input layer & bias
        self.w_input = np.random.uniform(-0.035, 0.035, (HIDDEN_LAYER_SIZE, self.train_set[0][0].shape[0]))
        self.b_input = np.array([[0.0]] * HIDDEN_LAYER_SIZE)

        # Weights of hidden layer & bias
        self.w_hidden = np.random.uniform(-0.1415, 0.1415, (self.num_of_classes, HIDDEN_LAYER_SIZE))
        self.b_hidden = np.array([[0.0]] * self.num_of_classes)

    def predict(self, record):
        values = self.forward_propagation(record)
        return np.argmax(values["hidden"]["outputs"])

    def forward_propagation(self, record):
        """
        Stream the data record in the network, saving each layer's input & output for back-prop
        :param record: data record
        :return: dictionary containing original record, and each layer's input & outputs values vectors
        """
        input_layer_inputs = np.dot(self.w_input, record) + self.b_input
        input_layer_outputs = reLU(input_layer_inputs)
        hidden_layer_inputs = np.dot(self.w_hidden, input_layer_outputs) + self.b_hidden
        hidden_layer_outputs = softmax(hidden_layer_inputs)
        return {"original": record,
                "input": {"inputs": input_layer_inputs, "outputs": input_layer_outputs},
                "hidden": {"inputs": hidden_layer_inputs, "outputs": hidden_layer_outputs}}

    def back_propagation(self, records_values):
        """
        Modify the network's weights using the Back-Prop method
        :param records_values: list of forward_propagation's dictionaries outputs, the size of the list is MINI_BATCH_SIZE at most
        :return: dictionary containing the gradients vectors calculated for each layer
        """
        # Initialize the gradients
        w_input_gradient = np.zeros(self.w_input.shape)
        b_input_gradient = np.zeros(self.b_input.shape)
        w_hidden_gradient = np.zeros(self.w_hidden.shape)
        b_hidden_gradient = np.zeros(self.b_hidden.shape)

        # For each record in current batch
        for record_values in records_values:
            # Calc the error gradient using the softmax distribution
            error_gradient = record_values["hidden"]["outputs"]
            error_gradient[record_values["correct class"]] -= 1

            # Sum up all gradients in hidden layer
            b_hidden_gradient += error_gradient
            w_hidden_gradient += error_gradient * record_values["input"]["outputs"].T

            # Sum up all gradients in input layer
            b_input_gradient += reLU_gradient(record_values["input"]["inputs"]) * np.dot(self.w_hidden.T, error_gradient)
            w_input_gradient += np.dot(
                reLU_gradient(record_values["input"]["inputs"]) * np.dot(self.w_hidden.T, error_gradient),
                record_values["original"].T)

        # Normalize to the sum of the batch's gradients to an average
        b_hidden_gradient /= len(records_values)
        w_hidden_gradient /= len(records_values)
        b_input_gradient /= len(records_values)
        w_input_gradient /= len(records_values)

        return {"input": {"weights": w_input_gradient, "bias": b_input_gradient},
                "hidden": {"weights": w_hidden_gradient, "bias": b_hidden_gradient}}

    def update_weights(self, gradients):
        """
        Update weights vectors using the gradients
        :param gradients: the dictionary returned by back-propagation, containing gradients vectors for each layer
        """
        self.b_input -= LEARNING_RATE * gradients["input"]["bias"]
        self.w_input -= LEARNING_RATE * gradients["input"]["weights"]
        self.b_hidden -= LEARNING_RATE * gradients["hidden"]["bias"]
        self.w_hidden -= LEARNING_RATE * gradients["hidden"]["weights"]

    def train(self):
        """
        Trains the Neural Network
        """

        # Split data to train & validation sets
        training_set = self.train_set[:int(TRAIN_VALIDATION_RATIO * len(self.train_set))]
        validation_set = self.train_set[int(TRAIN_VALIDATION_RATIO * len(self.train_set)):]

        # Init accuracies arrays
        training_accuracies = []
        validation_accuracies = []

        # Foreach epoch
        for epoch_number in range(NUM_OF_EPOCHS):
            np.random.shuffle(training_set)

            train_correct_predictions_counter = 0
            batch = []

            # Foreach record on train_set
            for record in training_set:
                # Propagate & save the records values at each step, including the correct class
                record_values = self.forward_propagation(record[0])
                record_values["correct class"] = record[1]
                batch.append(record_values)

                # Accurate prediction
                if np.argmax(record_values["hidden"]["outputs"]) == record_values["correct class"]:
                    train_correct_predictions_counter += 1

                # If the batch size reached, run back propagation
                if len(batch) == MINI_BATCH_SIZE:
                    self.update_weights(self.back_propagation(batch))
                    batch = []

            # For the remaining of un-propagated data records
            if len(batch) != 0:
                self.update_weights(self.back_propagation(batch))

            # Calc validation accuracy
            validation_correct_predictions_counter = 0
            for record in validation_set:
                if self.predict(record[0]) == record[1]:
                    validation_correct_predictions_counter += 1

            # Calc & append accuracies
            training_accuracies.append(float(train_correct_predictions_counter) / len(training_set))
            validation_accuracies.append(float(validation_correct_predictions_counter) / len(validation_set))

            print("Epoch number %d, training accuracy = %f, validation accuracy = %f" % (epoch_number,
                                                                                         training_accuracies[-1],
                                                                                         validation_accuracies[-1]
                                                                                         ))

        # Show accuracy graphs
        show_accuracy_graph(training_accuracies, "Training Accuracy Graph")
        show_accuracy_graph(validation_accuracies, "Validation Accuracy Graph")
        plt.savefig("Accuracies.png")


if __name__ == "__main__":
    # Normalize, shuffle & add classification to train_x
    train_set = list(zip(
        np.expand_dims(np.loadtxt("train_x", dtype=np.uint8), axis=2) / 255.0,
        np.loadtxt("train_y", dtype=np.uint8)
    ))
    np.random.shuffle(train_set)

    # Create & train the neural network
    neural_network = NeuralNetwork(NUM_OF_CLASSES, train_set)
    neural_network.train()

    # Run on test set
    test_set = np.expand_dims(np.loadtxt("test_x"), axis=2) / 255.0
    predictions = []
    for record in test_set:
        predictions.append(neural_network.predict(record))

    # Print test_y
    write_predictions("test_y", predictions)