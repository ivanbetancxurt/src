import torch as th

class NCA(th.nn.Module):
    def __init__(self, n_hidden_channels: int = 20, device: th.device = None) -> None:
        super().__init__()
        self.n_hidden_channels = n_hidden_channels
        self.n_channels = 10 + n_hidden_channels
        self.device = device or th.device('cpu')

    def encode(self, grid: th.LongTensor) -> th.FloatTensor:
        '''
            Converts ARC grid (2D tensor of values 0-9) into model-readable format (H x W x (10 + hidden channels)).
        '''
        one_hot_grid = th.nn.functional.one_hot(grid, num_classes=10)
        hidden_channels = th.zeros(grid.shape[0], grid.shape[1], self.n_hidden_channels)
        return th.cat((one_hot_grid, hidden_channels), dim=2)

    def decode(self, grid: th.FloatTensor) -> th.LongTensor:
        '''
            Converts an intermediary grid back into ARC format.
        '''
        color_channels = grid[:, :, :10]
        return th.argmax(color_channels, dim=2)