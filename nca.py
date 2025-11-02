import os
import torch as th
import numpy as np
from collections import defaultdict
import json
import random

class PerPixelLayerNorm(th.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.ln = th.nn.LayerNorm(normalized_shape=n_channels)
    
    def forward(self, x: th.FloatTensor):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class NCA(th.nn.Module):
    def __init__(self, n_hidden_channels: int = 20, out_channels: int = 64) -> None:
        super().__init__()
        self.n_hidden_channels = n_hidden_channels
        self.n_channels = 10 + n_hidden_channels
        self.perceive = th.nn.Conv2d(in_channels=self.n_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.normalizer = PerPixelLayerNorm(n_channels=out_channels)
        self.update = th.nn.Conv2d(in_channels=out_channels, out_channels=self.n_channels, kernel_size=1)

    def encode(self, grids: th.LongTensor) -> th.FloatTensor:
        '''
            Converts ARC grids (2D tensors of values 0-9) into model-readable format (N x (10 + hidden channels) x H x W).
        '''
        one_hot_grids = th.nn.functional.one_hot(grids, num_classes=10).float()
        hidden_channels = th.zeros(grids.shape[0], grids.shape[1], grids.shape[2], self.n_hidden_channels, device=grids.device)
        encoded_grids = th.cat((one_hot_grids, hidden_channels), dim=3)
        return encoded_grids.permute(0, 3, 1, 2)

    def decode(self, grid: th.FloatTensor) -> th.LongTensor:
        '''
            Converts an intermediary grid back into ARC format (N x H x W).
        '''
        return grid[:, :10].argmax(dim=1)

    def forward(self, x: th.FloatTensor) -> th.FloatTensor:
        '''
            Single forward pass of rules on a batch of grids, returning the updated states.
        '''
        y = self.perceive(x)
        y = self.normalizer(y)
        y = th.nn.functional.relu(y, inplace=False)
        return self.update(y)

    def async_update(self, prev_state: th.FloatTensor, proposed_state: th.FloatTensor, mask_prob):
        '''
            Mask the new state, with probability mask_prob, by interpolating its cells with the those of the previous state by a random stength.
        '''
        mask = th.rand_like(prev_state[:, :1, :, :]) < mask_prob
        strengths = th.rand_like(prev_state[:, :1, :, :])
        M = mask.float() * strengths
        return ((1 - M) * proposed_state) + (M * prev_state)

    def rollout(self, state: th.FloatTensor, steps: int, mask_prob_low: float = 0.0, mask_prob_high: float = 0.75, force_sync: bool = False) -> list[th.FloatTensor]:
        '''
            Applies 'steps' forward passes to the inputs and returns all the intermediate states.
        '''
        states = [state]
        mask_prob = 0.0 if force_sync else th.distributions.Uniform(mask_prob_low, mask_prob_high).sample().item()

        for _ in range(steps):
            proposed_state = self.forward(state)
            state = self.async_update(prev_state=state, proposed_state=proposed_state, mask_prob=mask_prob)
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

        step_losses = th.stack(step_losses)
        return (step_losses, step_losses.mean())

    def load_data(self, data_directory: str) -> list[dict]:
        '''
            Convert json data into python dictionaries.
        '''
        tasks = []
        for file in os.listdir(data_directory):
            with open(os.path.join(data_directory, file), 'r') as f:
                task = json.load(f)
                tasks.append(task)
        
        return tasks

    def train(self, data_directory: str, epochs: int = 10, learning_rate: float = 0.002, steps_per_batch: int = 10):
        '''
            Train model on given tasks.
        '''
        tasks = self.load_data(data_directory)
        
        shape_buckets = defaultdict(list)
        for task in tasks:
            for example in task['train']:
                shape_buckets[np.asarray(example['input']).shape].append(example)

        optimizer = th.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0001 / learning_rate, total_iters=epochs)

        for epoch in range(epochs):
            for example_list in shape_buckets.values():
                random.shuffle(example_list)

                inputs = th.stack([th.tensor(example['input'], dtype=th.long) for example in example_list])
                targets = th.stack([th.tensor(example['output'], dtype=th.long) for example in example_list])

                inputs = self.encode(inputs)
                states = self.rollout(inputs, steps=steps_per_batch)
                _, avg_loss = self.per_pixel_log_loss(states, targets)

                optimizer.zero_grad(set_to_none=True)
                avg_loss.backward()
                optimizer.step()

            scheduler.step()

    @th.no_grad()
    def evaluate(self, inputs: th.LongTensor, targets: th.LongTensor, steps: int = 20) -> dict:
        '''
            Evaluate learned rules on new data.
        '''
        self.eval()
        inputs = self.encode(grids=inputs)
        states = self.rollout(state=inputs, steps=steps, force_sync=True)

        accs = []
        for state in states:
            pred = self.decode(state)
            accs.append((pred == targets).float().mean().item())
        
        return {
            'final_accuracy': accs[-1],
            'per_step_accuracies': accs,
            'final_state': states[-1]
        }