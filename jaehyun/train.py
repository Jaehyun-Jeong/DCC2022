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

        self.step_done = 0

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

        mean_precision = 0
        mean_recall = 0
        mean_fscore = 0

        for data in test_loader:
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            test_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # To change it to numpy.ndarray
            labels = labels.cpu()
            predicted = predicted.cpu()
            precision, recall, fscore, _ = precision_recall_fscore_support(
                    labels.data,
                    predicted,
                    average='macro')

            mean_precision += precision
            mean_recall += recall
            mean_fscore += fscore

        mean_precision = mean_precision / len(test_loader)
        mean_recall = mean_recall / len(test_loader)
        mean_fscore = mean_fscore / len(test_loader)
        test_loss = test_loss / len(test_loader)

        accuracy = 100 * correct // total

        results = {
                'accuracy': accuracy,
                'precision': mean_precision,
                'recall': mean_recall,
                'f1-score': mean_fscore,
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

    # save class
    def save(self, saveDir: str = str(datetime)+".obj"):

        save_dict = self.__dict__

        # belows are impossible to dump
        save_dict.pop('tensorboardWriter', None)
        save_dict.pop('trainEnv', None)
        save_dict.pop('testEnv', None)
        save_dict.pop('device', None)

        # save model state dict
        save_dict['modelStateDict'] \
            = save_dict['model'].model.state_dict()
        save_dict['model'].model = None
        save_dict['optimizerStateDict'] \
            = save_dict['optimizer'].optimizer.state_dict()
        save_dict['optimizer'].optimizer = None

        torch.save(save_dict, saveDir)

    # Load class
    def load(self, loadDir: str):

        # Load torch model
        loadedDict = torch.load(loadDir, map_location=self.device)

        # Load state_dict of torch model, and optimizer
        try:

            self.model.load_state_dict(
                    loadedDict.pop('modelStateDict'))
            self.optimizer.load_state_dict(
                    loadedDict.pop('optimizerStateDict'))

            loadedDict.pop('modelStateDict')
            loadedDict.pop('optimizerStateDict')

        except ValueError:
            print(
                "No matching torch.nn.Module,"
                "please use equally shaped torch.nn.Module as you've done!")

        for key, value in loadedDict.items():
            self.__dict__[key] = value


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
