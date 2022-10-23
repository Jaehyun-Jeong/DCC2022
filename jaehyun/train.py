from typing import Dict

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
            scheduler=None,
            device: torch.device = torch.device('cpu'),
            ):

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def train(
            self,
            epochs: int,
            train_loader,
            test_loader):

        for epoch in range(epochs):  # loop over the dataset multiple times

            self.model.train()
            train_loss = 0

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

                # accmulate train loss
                train_loss += loss.item()

                if self.scheduler:
                    self.scheduler.step()

            train_loss = train_loss / len(train_loader)

            self.model.eval()

            results = self.test(test_loader)
            results['train loss'] = train_loss

            self.print_results(epoch, results)

    @torch.no_grad()
    def test(
            self,
            test_loader) -> Dict[str, float]:

        correct = 0
        total = 0

        test_loss = 0

        for data in test_loader:
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            test_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)

        accuracy = 100 * correct // total

        # To change it to numpy.ndarray
        labels = labels.cpu()
        predicted = predicted.cpu()
        precision, recall, fscore, _ = precision_recall_fscore_support(
                labels.data,
                predicted,
                average='macro')

        results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1-score': fscore,
                'test loss': test_loss}

        return results

    def print_results(
            self,
            epoch: int,
            results: Dict[str, float]):

        results_str = \
            f"| epoch:" \
            f" {str(epoch + 1)[0:10]:>10} " \

        for result_name, result in results.items():
            results_str += \
                f"| {result_name}:" \
                f" {str(result)[0:10]:>10} "

        results_str += "|"

        print(results_str)


if __name__ == "__main__":

    from model import Model
    from load import Load

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
