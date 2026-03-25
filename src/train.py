"""
Training loop and evaluation.
"""
import tensorflow as tf


def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
    """Compile and train the model."""
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # TODO: adjust for your task
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    return results
