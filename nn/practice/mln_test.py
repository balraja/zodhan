
from zodhan.nn.layered.mln import *
from zodhan.nn.utils.loss import SquaredLoss
from zodhan.nn.utils.activation import SigmoidFunction

import numpy as np

def mln_test():
    mln_builder = MLNBuilder(0.01)
    mln_builder.add_layer(3, 2, SigmoidFunction())
    mln_builder.add_layer(2, 3, SigmoidFunction())
    mln = mln_builder.build()
    input = np.random.rand(9).reshape(3, 3)
    output = np.eye(3)
    mln_trainer = MLNTrainer(mln, SquaredLoss(), input, output, 1, 2, 0.01)
    mln = mln_trainer.train()

if __name__ == "__main__":
    mln_test()