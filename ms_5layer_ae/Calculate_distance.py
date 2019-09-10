# Calculate distance for Single perceptron

import math


def MSE_b(SupervisedSignal, Output):
    delta0 = Output - SupervisedSignal
    #    delta0 = (Output - SupervisedSignal) * Output * (1 - Output)

    return delta0


def MSE(SupervisedSignal, Output):
    delta0 = ((SupervisedSignal - Output) ** 2) / 2

    return delta0


def CE_b(SupervisedSignal, Output):
    delta0 = Output - SupervisedSignal

    return delta0


def CE(SupervisedSignal, Output):
    delta0 = -(SupervisedSignal * math.log(Output) + (1 - SupervisedSignal) * math.log(1 - Output));

    return delta0


def IS_b(SupervisedSignal, Output):
    delta0 = (1 - (SupervisedSignal / Output)) * (1 - Output)

    return delta0


def IS(SupervisedSignal, Output):
    delta0 = (SupervisedSignal / Output) - math.log(SupervisedSignal / Output) - 1

    return delta0
