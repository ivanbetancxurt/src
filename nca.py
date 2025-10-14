import torch as th

class NCA(th.nn.Module):
    def __init__(self, n_hidden_channels: int = 20,) -> None:
        super().__init__()
        self.n_hidden_channels = n_hidden_channels
        self.n_channels = 10 + n_hidden_channels
        self.perceive = th.nn.Conv2d(in_channels=self.n_channels, out_channels=64, kernel_size=3, padding=1) #! hard-coded output channels
        self.norm = th.nn.GroupNorm(num_groups=1, num_channels=64) #! hard-coded output channels
        self.update = th.nn.Conv2d(in_channels=64, out_channels=self.n_channels, kernel_size=1) #! hard-coded input channels

    def encode(self, grids: th.LongTensor) -> th.FloatTensor:
        '''
            Converts ARC grids (2D tensors of values 0-9) into model-readable format (N x (10 + hidden channels) x H x W).
        '''
        one_hot_grids = th.nn.functional.one_hot(grids, num_classes=10).float()
        hidden_channels = th.zeros(grids.shape[0], grids.shape[1], grids.shape[2], self.n_hidden_channels, device=grids.device)
        encoded_grid = th.cat((one_hot_grids, hidden_channels), dim=3)
        return encoded_grid.permute(0, 3, 1, 2)

    def decode(self, grid: th.FloatTensor) -> th.LongTensor:
        '''
            Converts an intermediary grid back into ARC format (N x H x W).
        '''
        return grid[:, :10].argmax(dim=1)

    def forward(self, x: th.FloatTensor) -> th.FloatTensor:
        '''
            Single forward pass of rules on a grid, returning the updated state.
        '''
        y = self.perceive(x)
        y = self.norm(y)
        y = th.nn.functional.relu(y, inplace=False)
        return self.update(y)

    def rollout(self, state: th.FloatTensor, steps: int) -> list[th.FloatTensor]:
        '''
            Applies 'steps' forward passes to the input and returns all the intermediate states
        '''
        states = [state]
        for _ in range(steps):
            state = self.forward(state)
            states.append(state)
        return states

    def per_pixel_log_loss(self, states: list[th.FloatTensor], target: th.LongTensor) -> (th.FloatTensor, float):
        '''
            Compute CE loss per step and overall average.
        '''
        step_losses = []
        for state in states[1:]:
            logits = state[:, :10]
            step_losses.append(th.nn.functional.cross_entropy(input=logits, target=target))

        step_losses = th.tensor(step_losses, dtype=th.float)
        return (step_losses, step_losses.mean())