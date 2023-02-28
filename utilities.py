import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"G:\\Gachon Masters\\pythonKOA\\models\\b0\\b0{pretrained}.pth")


def save_plots(train_acc, valid_acc,test_acc, train_loss, valid_loss, test_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.plot(
        test_acc, color='red', linestyle='-',
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"G:\\Gachon Masters\\pythonKOA\\plots\\bo_accuracy{pretrained}.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.plot(
        test_loss, color='green', linestyle='-',
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"G:\\Gachon Masters\\pythonKOA\\plots\\bo_lossb0{pretrained}.png")
