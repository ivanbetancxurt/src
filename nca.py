import torch as th

class NCA(th.nn.Module):
    def __init__(self, n_hidden_channels: int = 20, device: th.device = None) -> None:
        super().__init__()
        self.n_hidden_channels = n_hidden_channels
        self.n_channels = 10 + n_hidden_channels
        self.device = device or th.device('cpu') #? this okay?
        self.perceive = th.nn.Conv2d(in_channels=self.n_channels, out_channels=64, kernel_size=3, padding=1) #! hard-coded output channels
        self.norm = th.nn.GroupNorm(num_groups=1, num_channels=64) #! hard-coded output channels
        self.update = th.nn.Conv2d(in_channels=64, out_channels=self.n_channels, kernel_size=1) #! hard-coded input channels

    def encode(self, grid: th.LongTensor) -> th.FloatTensor:
        '''
            Converts ARC grid (2D tensor of values 0-9) into model-readable format ((10 + hidden channels) x H x W).
        '''
        one_hot_grid = th.nn.functional.one_hot(grid, num_classes=10).to(self.device)
        hidden_channels = th.zeros(grid.shape[0], grid.shape[1], self.n_hidden_channels).to(self.device)
        encoded_grid = th.cat((one_hot_grid, hidden_channels), dim=2)
        return encoded_grid.permute(2, 0, 1)

    def decode(self, grid: th.FloatTensor) -> th.LongTensor:
        '''
            Converts an intermediary grid back into ARC format.
        '''
        color_channels = grid.permute(1, 2, 0)[:, :, :10]
        return th.argmax(color_channels, dim=2)