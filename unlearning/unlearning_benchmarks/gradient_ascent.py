import torch
from torch import optim
from torch import nn
from torch.optim import Adam, RMSprop, SGD
import timeit
from unlearning.unlearning_benchmarks import Unlearner
import copy


def eval_fool(model, intended_classes, target_loader, DEVICE):
    model.eval()
    for inputs, labels, index in target_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            output = model(inputs)
            predictions = torch.argmax(output.data, dim=1)
    if predictions[0] != intended_classes:
        # print("Target is not fooled ...")
        fooled = False
    else:
        # print("Target is fooled!")
        fooled = True
    return fooled * 1


def eval_model(model, loader, DEVICE: str):
    correct = 0.0
    model.eval()
    stable_losses = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels, idx = data
            model_device = next(model.parameters()).device
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            outputs = model(inputs)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
        acc = correct / len(loader.dataset)
        print(f'ACCURACY - {acc}')
    return acc


class MyOptimizer():
    ''' Defines a private optimizer object '''

    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer

    def step(self):
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class GA(Unlearner):
    """
    Provides model based on Gradient Ascent unlearning.
    """

    def __init__(self,
                 model,
                 lr: float = 5e-5,
                 epochs: int = 10,
                 lr_scheduler=None,
                 momentum: float = 0.9,
                 weight_decay: float = 5e-4,
                 device="cpu",
                 callback=None,
                 callback_epochs=None,
                 **kwargs) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """

        # self.config = config
        self.lr = lr
        self.epochs = epochs
        self.DEVICE = device
        self.momentum = momentum
        self.weight_decay = weight_decay

        if lr_scheduler is None:
            print("No LR scheduling is used")
            self.scheduler = lr_scheduler
        elif lr_scheduler == "Cosine-Annealing":
            raise NotImplementedError(
                "Cosine Annealing has not been implemented, yet.")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
        else:
            raise NotImplementedError(
                "This schedule has not been implemented, yet.")
        # copy of model
        import copy

        # Assuming model is your PyTorch model
        model_copy = copy.deepcopy(model)
        self.callback = callback
        self.callback_epochs = callback_epochs
        super(GA, self).__init__(model_copy)

    def get_updated_model(self,
                          retain_loader,
                          forget_loader,
                          validation_loader=None,
                          **kwargs):
        """
        Unlearning by fine-tuning.
        Args:
        net : nn.Module. pre-trained model to use as base of unlearning.
        retain_set : torch.utils.data.DataLoader.
            Dataset loader for access to the retain set. This is the subset
            of the training set that we don't want to forget.
        forget_set : torch.utils.data.DataLoader.
            Dataset loader for access to the forget set. This is the subset
            of the training set that we want to forget. This method doesn't
            make use of the forget set.
        validation_set : torch.utils.data.DataLoader.
            Dataset loader for access to the validation set. This method doesn't
            make use of the validation set.
        Returns:
        net : updated model
        """
        model_device = next(self.model.parameters()).device

        accs = {'test': [], 'train': [], 'forget': []}
        fooled_vec = {'epoch': [], 'step': []}
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        # myoptim = MyOptimizer(SGD(self.model.parameters(), lr=self.lr))
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        # continue training using gradient ascent on forget set
        start_time = timeit.default_timer()
        it = 0
        print(f"training for {self.epochs}")
        for epoch in range(self.epochs):
            print(f"training - {epoch}")
            for inputs, targets, idx in forget_loader:
                self.model.train()
                inputs, targets = inputs.to(model_device), targets.to(
                    model_device)

                # Convert inputs to the same dtype as the model's parameters
                dtype = next(self.model.parameters()).dtype
                inputs = inputs.to(dtype)

                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = (-1) * criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if validation_loader is not None:
                    print(f"validation_loader called ")
                    accs['test'].append(
                        eval_model(self.model, validation_loader,
                                   model_device))
                print(f"accs- {accs}")
                it += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if self.callback is not None:
                if epoch in self.callback_epochs:
                    self.callback(self.model, epoch)

        wall_clock_time = timeit.default_timer() - start_time
        # for last step
        if validation_loader is not None:
            accs['test'].append(
                eval_model(self.model, validation_loader, model_device))
        self.model.eval()
        return self.model, wall_clock_time, accs

    def __get_updated_model(self,
                            retain_loader,
                            forget_loader,
                            validation_loader=None,
                            **kwargs):
        """
        Unlearning by fine-tuning.
        Args:
          net : nn.Module. pre-trained model to use as base of unlearning.
          retain_set : torch.utils.data.DataLoader.
            Dataset loader for access to the retain set. This is the subset
            of the training set that we don't want to forget.
          forget_set : torch.utils.data.DataLoader.
            Dataset loader for access to the forget set. This is the subset
            of the training set that we want to forget. This method doesn't
            make use of the forget set.
          validation_set : torch.utils.data.DataLoader.
            Dataset loader for access to the validation set. This method doesn't
            make use of the validation set.
        Returns:
          net : updated model
        """
        model_device = next(self.model.parameters()).device

        accs = {'test': [], 'train': [], 'forget': []}
        fooled_vec = {'epoch': [], 'step': []}
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        # myoptim = MyOptimizer(SGD(self.model.parameters(), lr=self.lr))
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        # continue training using gradient ascent on forget set
        start_time = timeit.default_timer()
        it = 0
        print(f"training for {self.epochs}")
        for epoch in range(self.epochs):
            print(f"training - {epoch}")
            for inputs, targets, idx in forget_loader:

                self.model.train()
                inputs, targets = inputs.to(model_device), targets.to(
                    model_device)
                #inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = (-1) * criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if validation_loader is not None:
                    print(f"validation_loader called ")
                    accs['test'].append(
                        eval_model(self.model, validation_loader, self.DEVICE))
                print(f"accs- {accs}")
                it += 1
            if self.scheduler is not None:
                self.scheduler.step()

        wall_clock_time = timeit.default_timer() - start_time
        # for last step
        if validation_loader is not None:
            accs['test'].append(
                eval_model(self.model, validation_loader, self.DEVICE))
        # print('Evaluate model utility ...')
        # accs['test'] = eval_model(self.model, validation_set, self.DEVICE)
        # accs['train'] = eval_model(self.model, retain_set, self.DEVICE)
        # accs['forget'] = eval_model(self.model, forget_set, self.DEVICE)
        self.model.eval()
        return self.model, wall_clock_time, accs
