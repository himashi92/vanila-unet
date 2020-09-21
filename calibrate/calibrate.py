import fire
import sys
import os
import torch
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('..')

from data import SalObjDataset
from model.Unet_models import Sal_Feature_Nested_Unet


image_dir = '../data/tnbc/train/image/'
mask_dir = '../data/tnbc/train/mask/'

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1), logits.size(2), logits.size(3))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = torch.sigmoid(self.model(input))
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            labels_ex = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        # before_temperature_nll = nll_criterion(logits, labels.squeeze(1).long()).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        print('Before temperature - ECE: %.3f' % (before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        # optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        #
        # def eval():
        #     new_pred = self.temperature_scale(logits)
        #     loss = nll_criterion(new_pred, labels_ex.long())
        #     loss.backward()
        #     return loss
        # optimizer.step(eval)
        #
        # # Calculate NLL and ECE after temperature scaling
        # after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels_ex.long()).item()
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels_ex).item()
        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # softmaxes = F.softmax(logits)
        confidences, predictions = torch.max(logits, 1)
        accuracies = predictions.eq(labels.squeeze_(1))

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def demo(data='data', save='../models/Unet/tnbc/', depth=40, growth_rate=12, batch_size=1):

    # Load model state dict
    model_filename = os.path.join(save, 'Model_gen.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load validation indices
    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    valid_set = SalObjDataset(image_dir, mask_dir, 256)
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    orig_model = Sal_Feature_Nested_Unet().cuda()
    orig_model.load_state_dict(state_dict)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(orig_model)

    # Tune the model temperature, and save the results
    model.set_temperature(valid_loader)
    model_filename = os.path.join(save, 'model_with_temperature.pth')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model sved to %s' % model_filename)
    print('Done!')


if __name__ == '__main__':
    fire.Fire(demo)
