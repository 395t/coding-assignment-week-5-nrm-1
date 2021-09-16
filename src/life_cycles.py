import torch
from src.util.paths import CHECKPOINTS_DIR

from tqdm import tqdm

"""
Helper functions for things like training, testing, validating, saving models, loading models, etc. (things you
would do normally in the model testing phase)

Some functions are repurposed from https://github.com/395t/coding-assignment-week-4-opt-1/blob/main/notebooks/MomentumExperiments.ipynb
"""

def save_model(net: torch.nn.Module, name: str):
    torch.save(net, str(CHECKPOINTS_DIR / f'{name}.pt'))


def load_modal(name: str):
    return torch.load(str(CHECKPOINTS_DIR / f'{name}.pt'))


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(net: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion, trainloader, epochs: int = 10):
    device = get_device()
    net.train()

    metrics = {}

    for epoch in range(epochs):
        correct_images = 0
        total_images = 0
        training_loss = 0
        for batch_index, (images, labels) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            training_loss += loss.item()
            _, predicted = outputs.max(1)
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()

        epoch_metrics = {}
        epoch_metrics['correct_images'] = correct_images
        epoch_metrics['total_images'] = total_images
        epoch_metrics['loss'] = training_loss

        metrics[f'epoch_{epoch+1}'] = epoch_metrics

        print('Epoch: %d, Loss: %.3f, '
              'Accuracy: %.3f%% (%d/%d)' % (epoch, training_loss/(batch_index+1),
                                            100.*correct_images/total_images, correct_images, total_images))

        return metrics


def test_validation(net: torch.nn.Module, criterion, validloader):
    device = get_device()

    val_loss = 0
    total_images = 0
    correct_images = 0

    metrics = {}

    net.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(validloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()
    val_accuracy = 100.*correct_images/total_images

    metrics['loss'] = val_loss
    metrics['total_images'] = total_images
    metrics['correct_images'] = correct_images
    metrics['accuracy'] = val_accuracy

    #return val_loss/(batch_index+1), val_accuracy
    return metrics



def test(net: torch.nn.Module, criterion, testloader):
    device = get_device()

    metrics = {}

    test_loss = 0
    total_images = 0
    correct_images = 0
    net.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(testloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()
    test_accuracy = 100.*correct_images/total_images
    print("Loss on Test Set is", test_loss/(batch_index+1))
    print("Accuracy on Test Set is",test_accuracy)

    metrics['loss'] = test_loss
    metrics['total_images'] = total_images
    metrics['correct_images'] = correct_images

    return metrics


if __name__ == "__main__":
    from src.models.baselines.simple_cnn import SimpleCNN
    from src.models.baselines.vggnet import VGGStyleNet
    from src.models.baselines.convpool_cnn_c import ConvPoolCNNC
    from src.models.baselines.resnet import ResNet18

    from src.datasets.datasets import get_cifar_10

    # net = SimpleCNN()
    # net = VGGStyleNet()
    # net = ConvPoolCNNC()
    # net = ResNet18(num_classes=10)
    net = load_modal('test')

    trainloader, testloader, validloader = get_cifar_10(batch_size=256)

    LR = 0.001
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.999)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    criterior = torch.nn.CrossEntropyLoss()

    train(net, optimizer, criterior, trainloader, epochs=3)

    save_model(net, 'test')
