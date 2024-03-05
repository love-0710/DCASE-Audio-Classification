import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        keras.layers.LSTM(64),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model

def prepare_datasets(inputs, targets):
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.3)
    return x_train, x_test, x_validation, y_train, y_test, y_validation

def load_data(dataset):
    with open(dataset, "r") as fp:
        data = json.load(fp)
    inputs = np.array(data["mfcc"])
    targets = np.array(data['labels'])
    return inputs, targets

def plot_history(history):
    fig, ax = plt.subplots(2, figsize=(10, 8))

    # Plot Train Accuracy against Validation Accuracy
    ax[0].plot(history.history["accuracy"], label="Train Accuracy")
    ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(loc="lower right")
    ax[0].set_title("Accuracy Evaluation")

    ax[0].grid(True)

    # Add some space between the two graphs
    ax[0].set_ylim([0, 1.0])
    ax[0].set_yticks(np.arange(0, 1.1, step=0.1))

    # Plot Train Loss against Validation Loss
    ax[1].plot(history.history["loss"], label="Train Loss")
    ax[1].plot(history.history["val_loss"], label="Validation Loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Loss Evaluation")

    ax[1].grid(True)

    # Add some space between the two graphs
    ax[1].set_ylim([0, max(history.history["loss"])])
    ax[1].set_yticks(np.arange(0, max(history.history["loss"]), step=0.5))

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def main():
    DATA_FILE = "data.json"

    # Load Data
    inputs, targets = load_data(DATA_FILE)

    # Prepare Datasets
    x_train, x_test, x_validation, y_train, y_test, y_validation = prepare_datasets(inputs, targets)

    # Build Model
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    # Compile Model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train Model
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=50, batch_size=32)

    # Evaluate
    plot_history(history)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
