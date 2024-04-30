import torch
from torch import nn
import time
import numpy as np
import matplotlib.pyplot as plt
import os

from model import MyCNN
from dataset import ImagesDataset

def train_test_split(x, y, split_ratio, seed = 0):
    torch.manual_seed(seed)
    indices = torch.randperm(x.size(0))
    split_index = int(x.size(0)*split_ratio)
    x_train = x[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_train = y[indices[:split_index]]
    y_test = y[indices[split_index:]]
    return x_train, x_test, y_train, y_test

def get_batch(x, y, batch_size):
    """
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    """
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i : i + batch_size, :, :, :]
        batch_y = y[i : i + batch_size]
        yield (batch_x, batch_y)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



def train(args, cnn=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    # TODO: necessary?
    torch.set_num_threads(5)
    # Numpy random seed
    #npr.seed(args.seed)

    # Save directory
    save_dir = "outputs/" + args.experiment_name

    # INPUT CHANNEL
    # LOAD THE MODEL
    if cnn is None:
        cnn = MyCNN()

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)


    # Create the outputs folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Beginning training ...")
    if torch.cuda.is_available():
        cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        cnn.train()  # Change model to 'train' mode
        losses = []
        for images, _, labels in args.train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        # plot training images
        #if args.plot:
        #    _, predicted = torch.max(outputs.data, 1, keepdim=True)
        #    plot(
        #        xs,
        #        ys,
        #        predicted.cpu().numpy(),
        #        colours,
        #        save_dir + "/train_%d.png" % epoch,
        #        args.visualize,
        #        args.downsize_input,
        #    )

        # plot training images
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print(
            "Epoch [%d/%d], Loss: %.4f, Time (s): %d"
            % (epoch + 1, args.epochs, avg_loss, time_elapsed)
        )

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.0
        total = 0.0
        losses = []
        for images, _, labels in args.val_loader:

            outputs = cnn(images)
            val_loss = criterion(outputs, labels)
            losses.append(val_loss.data.item())

            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        val_loss = np.mean(losses)
        val_acc = 100 * correct / total
        
    

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print(
            "Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %.2f"
            % (epoch + 1, args.epochs, val_loss, val_acc, time_elapsed)
        )

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")

    if args.checkpoint:
        print("Saving model...")
        torch.save(cnn.state_dict(), args.checkpoint)

    return cnn




if __name__ == "__main__":
    pass
    #profiles_dataset = ImagesDataset(csv_file="data/input/profiles_pics.csv")
    #images_all = torch.stack([profiles_dataset[i][0] for i in range(len(profiles_dataset))])
    #age_classes_all = torch.tensor([profiles_dataset[i][2] for i in range(len(profiles_dataset))], dtype=torch.long)
#
    #images_train, images_test, ages_train, ages_test = train_test_split(images_all, age_classes_all, 0.8)
#
    #args = AttrDict()
    #args_dict = {
    #    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #    "checkpoint": "",
    #    'learn_rate':0.001, 
    #    "batch_size": 100,
    #    "epochs": 5,
    #    #"seed": 0,
    #    "plot": True,
    #    "experiment_name": "age_classification",
    #    "visualize": False,
    #    "downsize_input": False,
    #    "train_loader": images_all, 
    #    "val_loader": images_all 
    #}
    #args.update(args_dict)
    #cnn = train(args)