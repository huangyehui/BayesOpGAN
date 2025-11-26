import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize



class BayesianMode():

    def __init__(self):
        
        return
        


    def fit(self, train_X, train_Y):
        self.bounds = torch.stack([torch.zeros(train_X.size(1)), torch.ones(train_X.size(1))]).cuda()
        # print("bound size={}".format(self.bounds.size()))
        # print("bound ={}".format(self.bounds))
        # print("train_X size={}".format(train_X.size()))
        gp = SingleTaskGP(train_X, train_Y,
                          input_transform=Normalize(d=256),
    outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        return gp

    def ac_func(self, gp):
        UCB = UpperConfidenceBound(gp, beta=0.1)
        candidate, acq_value = optimize_acqf(
            UCB, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
        )
        return candidate




