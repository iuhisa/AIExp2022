'''
CAEModel
'''
from . import init_weights
from .networks import CAE

class CAEModel():
    def __init__(self):
        self.net = CAE()
        self.net.apply(init_weights)

        # optimizer
        # DataParallel
        # 
    def set_input():
        pass

    def forward():
        pass

    def backward():
        pass

    def optimize():
        pass

    def get_loss():
        pass

    def save():
        pass

    def load():
        pass

    def train():
        pass

    def eval():
        pass

