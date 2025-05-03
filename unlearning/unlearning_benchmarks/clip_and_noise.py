import torch

from torch import optim
from torch import nn
from torch.optim import Adam, RMSprop, SGD
import timeit

from unlearning.unlearning_benchmarks import Unlearner

# for efficient computations of per-sample-gradients
from opacus.grad_sample import GradSampleModule
from opacus.validators import ModuleValidator
from opacus.validators import register_module_validator


def eval_model(model, loader, DEVICE: str):
    correct = 0.0
    model.eval()
    stable_losses = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels, idx = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
        acc = correct / len(loader.dataset)
        print(f'ACCURACY - {acc}')
    return acc


class PrivateOptimizer():
    ''' Defines a private optimizer object '''

    def __init__(self, base_optimizer, model, C=float("inf"), tau=0.0):
        self.model = model
        self.base_optimizer = base_optimizer
        self.C = C
        self.tau = tau
        self.mygradlist = []
        self.grad_sum = 0.0
        self.batch_len = 0.0

    def aggregate_crop_grads(self, model):
        """ Aggregate gradients of a subbatch and store them in this object. """
        grad_sum = 0.0
        with torch.no_grad():
            for t in model.parameters():
                if t.requires_grad == False:
                    continue
                if t.grad_sample is not None:
                    grad_sum += torch.sum(t.grad_sample.reshape(
                        len(t.grad_sample), -1).pow(2),
                                          dim=1)
        self.batch_len += len(grad_sum)
        target_mult = torch.ones_like(grad_sum)
        target_mult[grad_sum > self.C * self.C] = self.C / torch.sqrt(
            grad_sum[grad_sum > self.C * self.C])

        self.grad_sum += torch.sum(grad_sum.detach())
        with torch.no_grad():
            for pidx, t in enumerate(model.parameters()):
                if t.requires_grad == False:
                    if len(self.mygradlist) <= pidx:
                        self.mygradlist.append([])
                    continue
                # Crop and store per microbatch sum
                cropgrad = torch.sum(
                    target_mult.reshape([len(target_mult)] + [1] *
                                        (len(t.grad_sample.shape) - 1)) *
                    t.grad_sample, 0).detach()
                if len(self.mygradlist) <= pidx:
                    self.mygradlist.append([cropgrad])
                else:
                    self.mygradlist[pidx].append(cropgrad)

    def step(self, model):
        # Performing updates
        grad_list = []
        with torch.no_grad():
            for i, (torg, t) in enumerate(
                    zip([
                        p
                        for p in self.base_optimizer.param_groups[0]["params"]
                    ], [p for p in model.parameters()])):
                if t.requires_grad == False:
                    continue
                if t.grad_sample is not None:
                    all_grad = torch.stack(self.mygradlist[i],
                                           dim=0).sum(dim=0) / self.batch_len
                    torg.grad = all_grad + self.tau * torch.randn(
                        all_grad.shape[1:], device=all_grad.device)
                    grad_list.append(torg.grad.clone())
        self.base_optimizer.step()
        return self.grad_sum, grad_list

    def zero_subbatch_grad(self, model):
        for t in model.parameters():
            t.grad_sample = None
            t.grad_summed = None
            self.base_optimizer.zero_grad()

    def zero_grad(self):
        self.batch_len = 0
        self.mygradlist = []
        self.base_optimizer.zero_grad()
        self.grad_sum = 0.0


class CaN(Unlearner):
    """
    Provides updated model based on Gradient Ascent / Descent subject to clipped and noised gradients.
    """

    def __init__(self,
                 model,
                 lr: float = 5e-5,
                 epochs: int = 1,
                 lr_scheduler=None,
                 loss_sign: int = 1,
                 tau: float = 0.001,
                 C: float = torch.inf,
                 momentum: float = 0.9,
                 weight_decay: float = 5e-4,
                 device="cpu") -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
            loss sign (int): either -1 or +1 
        """

        self.lr = lr
        self.epochs = epochs
        self.DEVICE = device
        self.C = C
        self.tau = tau
        self.loss_sign = loss_sign
        self.MAX_PHYSICAL_BATCH = 256

        super(CaN, self).__init__(model)

    def get_updated_model(self,
                          retain_set,
                          forget_set,
                          validation_set,
                          overwrite_dict=None,
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
          loader: either retain_loader or forget_loader; make sure to choose loss_sign accordingly
        Returns:
          net : updated model
        """
        if overwrite_dict is not None:
            self.C = overwrite_dict['C']
            self.tau = overwrite_dict['tau']
            self.lr = overwrite_dict['lr']
            self.epochs = overwrite_dict['epochs']
            self.loss_sign = overwrite_dict['loss_sign']

        accs = {}

        # register private optimizer
        myoptim = PrivateOptimizer(SGD(self.model.parameters(), lr=self.lr),
                                   self.model,
                                   C=self.C,
                                   tau=self.tau)
        criterion = nn.CrossEntropyLoss()

        if not isinstance(self.model, GradSampleModule):
            # make sure we can use Opacus's per-sample gradient interface
            if not ModuleValidator.is_valid(self.model):
                # this fixes classifier if necessary: https://opacus.ai/tutorials/building_image_classifier
                self.model = ModuleValidator.fix(self.model)
            self.model = GradSampleModule(
                self.model, loss_reduction="mean").to(self.DEVICE)

        if self.loss_sign == 1:
            loader = retain_set
        else:
            loader = forget_set

        start_time = timeit.default_timer()
        for epoch in range(0, self.epochs):
            running_loss = 0.0
            correct = 0
            grad_sum2 = 0.0
            self.model.train()
            num_samples = 0
            for i, data in enumerate(loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, idx = data
                inputs = inputs.to(self.DEVICE)
                labels = labels.to(self.DEVICE)
                ## Split the batch in physical batches.
                myoptim.zero_grad()
                outputs_list = []
                for subbatch_start in range(0, len(inputs),
                                            self.MAX_PHYSICAL_BATCH):
                    myoptim.zero_subbatch_grad(self.model)
                    subbatch_end = min(
                        subbatch_start + self.MAX_PHYSICAL_BATCH, len(inputs))
                    input_batch = inputs[subbatch_start:subbatch_end].clone()
                    labels_batch = labels[subbatch_start:subbatch_end].clone()
                    # forward + backward + optimize
                    outputs_batch = self.model(input_batch)
                    # Note that loss should return one element per batch item.
                    loss_batch = self.loss_sign * criterion(
                        outputs_batch, labels_batch)
                    loss_batch.backward()
                    myoptim.aggregate_crop_grads(self.model)
                    running_loss += loss_batch.item()
                    outputs_list.append(outputs_batch.detach())
                outputs = torch.cat(outputs_list)
                grad_sum_increment, _ = myoptim.step(self.model)
                grad_sum2 += grad_sum_increment
                num_samples += len(inputs)
                # Get statistics
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
            if not epoch % 10:
                print('Training [%d] loss: %.5f; acc: %.5f; grad_sum: %.5f' %
                      (epoch + 1, running_loss / len(loader.dataset),
                       correct / len(loader.dataset),
                       torch.sqrt(grad_sum2 / len(loader.dataset))))
            if self.scheduler is not None:
                self.scheduler.step()

        # evaluate model quality
        wall_clock_time = timeit.default_timer() - start_time
        print('Evaluate model utility ...')
        accs['test'] = eval_model(self.model, validation_set, self.DEVICE)
        accs['train'] = eval_model(self.model, retain_set, self.DEVICE)
        accs['forget'] = eval_model(self.model, forget_set, self.DEVICE)
        self.model.eval()
        return self.model, wall_clock_time, accs
