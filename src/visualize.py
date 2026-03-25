"""
Visualization and plotting utilities.
"""
import matplotlib.pyplot as plt
import os


FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def plot_training_history(history, save=True):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'training_history.png'), dpi=150)
        print(f"Saved to {FIGURES_DIR}/training_history.png")

    plt.show()
