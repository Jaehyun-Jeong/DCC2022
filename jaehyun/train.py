from typing import Dict

from model import Model
from load import Load

from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.optim as optim
from torch import nn
import torchvision.transforms as transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer():

    def __init__(
            self,
            model: nn.Module,
            optimizer,
            criterion,
            device: torch.device = torch.device('cpu'),
            ):

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(
            self,
            epochs: int,
            train_loader,
            test_loader):

        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.test(test_loader)

    @torch.no_grad()
    def test(
            self,
            test_loader) -> Dict[str, float]:

        correct = 0
        total = 0

        for data in test_loader:
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct // total
        precision, recall, fscore = precision_recall_fscore_support(
                labels.data,
                predicted,
                average='macro')

        results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1-score': fscore}

        return results

    def print_results(
            self,
            epoch: int,
            results: Dict[str, float]):

        results_str = f"| epoch: {epoch + 1} "
        for result_name, result in results.items():
            results_str += f"| {result_name}: {result} "
        results_str += "|"

        print(results_str)


if __name__ == "__main__":
    # Use Gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()

    transformer = [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize((32, 32))]

    # Train loader
    train_dataloader = Load(
            transformer,
            batch_size=128)
    _, trainloader = train_dataloader("./train_val_test_dataset/train")

    # Validation loader
    valid_dataloader = Load(
            transformer,
            batch_size=128)
    _, valloader = valid_dataloader("./train_val_test_dataset/valid")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Init Trainer
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device)

    # Train
    trainer.train(
            50,
            trainloader,
            valloader)
