import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
#from ....evaluators import ThresholdMIA

from unlearning.unlearning_benchmarks import Unlearner
from unlearning.unlearning_benchmarks import benchmarks

from torch import optim
from torch import nn
import timeit
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


class GD(Unlearner):
    """
    Provides model based on gradient descent unlearning.
    """

    def __init__(self,
                 model,
                 lr: float = 5e-5,
                 epochs: int = 10,
                 lr_scheduler=None,
                 momentum: float = 0.9,
                 weight_decay: float = 5e-4,
                 noise_var: float = 0.000,
                 device: str = "cpu",
                 callback=None,
                 callback_epochs=None,
                 **kwargs) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        print(f"learning rate is {lr}")
        #self.config = config
        self.lr = lr
        self.noise_var = noise_var
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.weight_decay = weight_decay

        model_copy = copy.deepcopy(model)
        self.model = model_copy

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        self.DEVICE = device
        if self.lr_scheduler is None:
            print("No LR scheduling is used")
            self.lr_scheduler = lr_scheduler
        elif self.lr_scheduler == "Cosine-Annealing":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)
        else:
            raise NotImplementedError(
                "This schedule has not been implemented, yet.")

        self.callback = callback
        self.callback_epochs = callback_epochs
        #super(GD, self).__init__(model_copy)

    def _eval_model(self, model, loader, acc_label=""):
        correct_predictions = 0.0
        model_device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            for inputs, labels, idx in loader:
                inputs, labels = inputs.to(model_device), labels.to(
                    model_device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / len(loader.dataset)
        # Optionally print the accuracy for debugging
        # print(f'{acc_label} ACCURACY - {accuracy:.4f}')

        return accuracy

    def get_updated_model(self,
                          retain_loader,
                          validation_loader=None,
                          forget_loader=None,
                          eval_epochs=None,
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
            wall_clock_time : time taken to update model
            accs : accuracies across train, test, forget points
        
        """
        print("Start training ...")
        print(f"val_loader - {validation_loader}")
        print(f"weight decay is : {self.weight_decay}")

        model_device = next(self.model.parameters()).device

        accs = {'test': [], 'retain': [], 'forget': []}
        tmias = {'scores': []}
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        # Continue training using gradient descent on retain set
        start_time = timeit.default_timer()
        it = 0
        fooled_vec = {'epoch': [], 'step': []}
        print(f"training for {self.epochs}")
        # Print number of points in retain_loader
        print(f"retain_loader - {len(retain_loader.dataset)}")

        for epoch in range(self.epochs):
            print(f"training - {epoch} / {self.epochs}")
            epoch_loss = 0  # Initialize epoch loss
            for inputs, targets, idx in retain_loader:
                self.model.train()
                inputs, targets = inputs.to(model_device), targets.to(
                    model_device)

                # Convert inputs to the same dtype as the model's parameters
                dtype = next(self.model.parameters()).dtype
                inputs = inputs.to(dtype)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                if self.noise_var > 0:
                    # Get flatten param vector
                    param_vector = parameters_to_vector(
                        self.model.parameters())
                    # Add noise to param vector
                    noise = torch.randn(
                        len(param_vector)).to(model_device) * torch.sqrt(
                            torch.tensor(self.noise_var)).to(model_device)
                    param_vector.add_(noise)
                    # Load params back to model
                    vector_to_parameters(param_vector, self.model.parameters())

                # Update the model parameters using the optimizer
                self.optimizer.step()

                epoch_loss += loss.item()  # Accumulate loss for the epoch

                it += 1

            # Print average loss for the epoch
            print(f"Epoch {epoch} Loss: {epoch_loss / len(retain_loader)}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.callback is not None:
                if epoch in self.callback_epochs:
                    self.callback(self.model, epoch)

            if eval_epochs is not None and epoch in eval_epochs:
                if validation_loader is not None:
                    accs['test'].append(
                        self._eval_model(self.model,
                                         validation_loader,
                                         acc_label="test"))
                if forget_loader is not None:
                    accs['forget'].append(
                        self._eval_model(self.model,
                                         forget_loader,
                                         acc_label="forget"))
                if retain_loader is not None:
                    accs['retain'].append(
                        self._eval_model(self.model,
                                         retain_loader,
                                         acc_label="retain"))

                print(
                    f"Epoch {epoch} - retain: {accs['retain'][-1]}, test: {accs['test'][-1]}, forget: {accs['forget'][-1]}"
                )

        wall_clock_time = timeit.default_timer() - start_time

        # Evaluate at the end if eval_epochs is None
        if eval_epochs is None:
            if validation_loader is not None:
                accs['test'].append(
                    self._eval_model(self.model, validation_loader))
            if forget_loader is not None:
                accs['forget'].append(
                    self._eval_model(self.model, forget_loader))
            if retain_loader is not None:
                accs['retain'].append(
                    self._eval_model(self.model, retain_loader))

        self.model.eval()

        return self.model, wall_clock_time, accs

    def __get_updated_model(self,
                            retain_loader,
                            validation_loader=None,
                            forget_loader=None,
                            eval_epochs=None,
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
        wall_clock_time : time taken to update model
        accs : accuracies across train, test, forget points
            
        NOTE: 
            currently 
                "mia_forget_loader" and "mia_test_loader" are not called

        """
        print("Start training ...")
        print(f"val_loader - {validation_loader}")
        print(f"weight decay is : {self.weight_decay}")

        model_device = next(self.model.parameters()).device

        accs = {'test': [], 'retain': [], 'forget': []}
        tmias = {'scores': []}
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        # continue training using gradient descent on retain set
        start_time = timeit.default_timer()
        it = 0
        fooled_vec = {'epoch': [], 'step': []}
        print(f"training for {self.epochs}")
        # print num points in retain_loader
        print(f"retain_loader - {len(retain_loader.dataset)}")

        for epoch in range(self.epochs):
            print(f"training - {epoch} / {self.epochs}")
            for inputs, targets, idx in retain_loader:
                self.model.train()
                inputs, targets = inputs.to(model_device), targets.to(
                    model_device)

                # Convert inputs to the same dtype as the model's parameters
                dtype = next(self.model.parameters()).dtype
                inputs = inputs.to(dtype)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if self.noise_var > 0:
                    # get flatten param vector
                    param_vector = parameters_to_vector(
                        self.model.parameters())
                    # add noise to param vector
                    noise = torch.randn(
                        len(param_vector)).to(model_device) * torch.sqrt(
                            torch.tensor(self.noise_var)).to(model_device)
                    param_vector.add_(noise)
                    # load params back to model
                    vector_to_parameters(param_vector, self.model.parameters())
                self.optimizer.step()
                # add adv eval: this one is cheap to compute!
                it += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.callback is not None:
                if epoch in self.callback_epochs:
                    self.callback(self.model, epoch)
            if epoch in eval_epochs:
                accs['test'].append(
                    self._eval_model(self.model,
                                     validation_loader,
                                     acc_label="test"))
                accs['forget'].append(
                    self._eval_model(self.model,
                                     forget_loader,
                                     acc_label="forget"))
                accs['retain'].append(
                    self._eval_model(self.model,
                                     retain_loader,
                                     acc_label="retain"))
                # print the last epoch for each
                print(
                    f"Epoch {epoch} - retain: {accs['retain'][-1]}, test: {accs['test'][-1]}, forget: {accs['forget'][-1]}"
                )

        wall_clock_time = timeit.default_timer() - start_time
        # for last step
        if eval_epochs is None:
            if validation_loader is not None:
                accs['test'].append(
                    self._eval_model(self.model, validation_loader))
            if forget_loader is not None:
                accs['forget'].append(
                    self._eval_model(self.model, forget_loader))
            if retain_loader is not None:
                accs['retain'].append(
                    self._eval_model(self.model, retain_loader))
        self.model.eval()
        # evaluate updated model
        # accs['train'] = self._eval_model(self.model, retain_set)
        # accs['test'] = self._eval_model(self.model, validation_set)
        # accs['forget'] = self._eval_model(self.model, forget_set)
        return self.model, wall_clock_time, accs
