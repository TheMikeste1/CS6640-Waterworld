import torch.nn


class MemoryLSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hx = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, self.hx = super().forward(input, self.hx)
        return output

    def reset(self):
        self.hx = None

    def train(self, mode: bool = True):
        self.reset()
        return super().train(mode)
