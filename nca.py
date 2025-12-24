from copy import deepcopy
import os
import torch as th
import numpy as np
from collections import defaultdict
import json
import random
from utils import progress_bar

class PerPixelLayerNorm(th.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.ln = th.nn.LayerNorm(normalized_shape=n_channels)
    
    def forward(self, x: th.FloatTensor):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class NCA(th.nn.Module):
    def __init__(self, n_hidden_channels: int = 20, temperature: int = 5) -> None:
        super().__init__()
        self.n_hidden_channels = n_hidden_channels
        self.n_channels = 10 + n_hidden_channels
        self.conv = th.nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=3, padding=1)
        self.normalizer1, self.normalizer2 = PerPixelLayerNorm(n_channels=self.n_channels), PerPixelLayerNorm(n_channels=self.n_channels)
        self.linear1 = th.nn.Conv2d(in_channels=(10 + self.n_channels), out_channels=self.n_channels, kernel_size=1)
        self.linear2 = th.nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=1)
        self.temperature = temperature

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

    def forward(self, grids: th.FloatTensor) -> th.FloatTensor:
        '''
            Single forward pass of rules on a batch of grids, returning the updated states. Must encode first if not running via rollout().
        '''
        initial_grids = grids
        grids = self.conv(grids)
        grids = self.normalizer1(grids)
        grids = th.nn.functional.relu(grids, inplace=False)

        grids = th.cat((grids, initial_grids[:, :10]), dim=1)
        grids = self.linear1(grids)
        grids = self.normalizer2(grids)
        grids = th.nn.functional.relu(grids, inplace=False)
        grids = self.linear2(grids)

        grids = th.cat((th.softmax(grids[:, :10] / self.temperature, dim=1), grids[:, 10:]), dim=1)

        return grids 

    def async_update(self, prev_state: th.FloatTensor, proposed_state: th.FloatTensor, mask_prob):
        '''
            Mask the new state, with probability mask_prob, by interpolating its cells with the those of the previous state by a random stength.
        '''
        mask = th.rand_like(prev_state[:, :1, :, :]) < mask_prob
        strengths = th.rand_like(prev_state[:, :1, :, :])
        M = mask.float() * strengths
        return ((1 - M) * proposed_state) + (M * prev_state)

    def rollout(self, state: th.FloatTensor, steps: int, mask_prob_low: float, mask_prob_high: float, force_sync: bool) -> list[th.FloatTensor]:
        '''
            Applies 'steps' forward passes to the inputs and returns all the intermediate states.
        '''
        state = self.encode(state)

        states = [state]
        mask_prob = 0.0 if force_sync else th.empty(1, device=state.device).uniform_(mask_prob_low, mask_prob_high).item()

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
            probs = state[:, :10].clamp_min(1e-8)
            step_losses.append(th.nn.functional.nll_loss(input=probs.log(), target=target))

        step_losses = th.stack(step_losses)
        return (step_losses, step_losses.mean())

    def load_task(self, task_path: str) -> dict:
        '''
            Convert an individual json task into a python dictionary.
        '''
        with open(task_path, 'r') as f:
            task = json.load(f)
        
        return task

    def load_data(self, data_directory: str) -> list[dict]:
        '''
            Convert json data into python dictionaries.
        '''
        tasks = []
        for file in os.listdir(data_directory):
            tasks.append(self.load_task(os.path.join(data_directory, file)))
        
        return tasks

    def get_shape_buckets(self, tasks: list[dict]) -> defaultdict[list]:
        '''
            Returns a dictionary of the training inputs of the given tasks grouped by their shape.
        '''
        shape_buckets = defaultdict(list)

        for task in tasks:
            for example in task['train']:
                shape_buckets[np.asarray(example['input']).shape].append(example)

        return shape_buckets

    def train_on_examples(
        self, 
        examples: list,
        steps: int,
        trials: int,
        mask_prob_low: float,
        mask_prob_high: float, 
        force_sync: bool,
        device
    ) -> float:
        '''
            Train NCA on a collection of examples.
        '''
        random.shuffle(examples)

        losses = []
        for _ in range(trials):
            inputs = th.stack([th.tensor(example['input'], dtype=th.long, device=device) for example in examples])
            targets = th.stack([th.tensor(example['output'], dtype=th.long, device=device) for example in examples])

            states = self.rollout(inputs, steps=steps, mask_prob_low=mask_prob_low, mask_prob_high=mask_prob_high, force_sync=force_sync)
            _, loss = self.per_pixel_log_loss(states, targets)
            losses.append(loss)

        avg_loss = th.stack(losses).mean()
        avg_loss.backward()
        return avg_loss.item()

    def fit(
        self, 
        data_directory: str, 
        epochs: int = 800, 
        steps: int = 10, 
        trials: int = 128, 
        learning_rate: float = 0.002,
        mask_prob_low: float = 0.0, 
        mask_prob_high: float = 0.75, 
        force_sync: bool = False
    ) -> list[float]:
        '''
            Train NCA on all tasks.
        '''
        tasks = self.load_data(data_directory)
        device = next(self.parameters()).device
        
        shape_buckets = self.get_shape_buckets(tasks)

        optimizer = th.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0001 / learning_rate, total_iters=epochs)

        print('==> Training...')
        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = []

            for i, example_list in enumerate(shape_buckets.values()):
                optimizer.zero_grad(set_to_none=True)
                progress_bar(i, len(shape_buckets.values()), f'Epoch: {epoch + 1}')
                
                avg_loss = self.train_on_examples(
                    examples=example_list, 
                    steps=steps, 
                    trials=trials, 
                    mask_prob_low=mask_prob_low, 
                    mask_prob_high=mask_prob_high,
                    force_sync=force_sync,
                    device=device
                )

                batch_losses.append(avg_loss)
                optimizer.step()

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            scheduler.step()
        
        return epoch_losses

    def lexi_fit(
        self, 
        data_directory: str,
        epsilon: float,
        epochs: int = 200, #! ATTENTION
        steps: int = 10, 
        trials: int = 128, 
        learning_rate: float = 0.002,
        mask_prob_low: float = 0.0, 
        mask_prob_high: float = 0.75, 
        force_sync: bool = False,
        pop_size: int = 4,
        use_sgd: bool = True,
    ):
        '''
            Train NCA on all tasks with gradient lexicase selection.
        '''
        def cosine_annealing_lr(epoch: int, lr: float = 0.1, T_max=epochs, eta_min=0):
            return eta_min + (0.5 * (lr - eta_min) * (1 + np.cos((epoch)/T_max * np.pi)))

        def subset_gd(epoch: int, children: list[NCA]) -> list[NCA]:
            '''
                Train each child NCA on a disjoint subset of the data.
            '''
            optimizers, schedulers = [], []
            for child in children:
                if use_sgd:
                    optimizer = th.optim.SGD(child.parameters(), lr=cosine_annealing_lr(epoch), momentum=0.9, weight_decay=1e-4)
                    optimizers.append(optimizer)
                else:
                    optimizer = th.optim.AdamW(child.parameters(), lr=learning_rate)
                    scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0001 / learning_rate, total_iters=epochs)
                    optimizers.append(optimizer)
                    schedulers.append(scheduler)
            
            task_indices = list(range(len(tasks)))
            random.shuffle(task_indices)
            task_idx_partitions = [task_indices[i::pop_size] for i in range(pop_size)]

            for i, child in enumerate(children):
                subset_tasks = [tasks[j] for j in task_idx_partitions[i]]
                shape_buckets = child.get_shape_buckets(subset_tasks)

                for j, example_list in enumerate(shape_buckets.values()):
                    optimizers[i].zero_grad(set_to_none=True)
                    progress_bar(j, len(shape_buckets.values()), f'Epoch {epoch + 1}: Training child {i}')
                    
                    avg_loss = child.train_on_examples(
                        examples=example_list,
                        steps=steps,
                        trials=trials, #! ATTENTION
                        mask_prob_low=mask_prob_low, 
                        mask_prob_high=mask_prob_high,
                        force_sync=force_sync,
                        device=device
                    )

                    optimizers[i].step()

                if not use_sgd: schedulers[i].step()

            return children
        
        def select(children: list[NCA]) -> list[NCA]:
            '''
                Select a child NCA with lexicase selection and generate a new population.
            '''
            pool = list(range(len(children)))
            print(f'==> Starting population size: {len(pool)}')

            cases = []
            for task in tasks:
                for example in task['train']:
                    cases.append(example)
            
            random.shuffle(cases)
            
            for case in cases:
                scores = []
                x = th.tensor(case['input'])
                y = th.tensor(case['output'])

                for child_idx in pool:
                    score = children[child_idx].evaluate(inputs=x.unsqueeze(0), targets=y.unsqueeze(0), steps=steps)['pixel_final_accuracy']
                    scores.append(score)
                
                best = max(scores)
                pool = [child_idx for (child_idx, score) in zip(pool, scores) if score >= best - epsilon]
                print(f'==> {len(pool)} remaining...')
                if len(pool) == 1: break
                    
            if len(pool) > 1:
                parent_idx = random.choice(pool)
            else:
                parent_idx = pool[0]
            
            parent = children[parent_idx]
            children = [deepcopy(parent) for _ in range(pop_size)]
            return children

        epochs *= (pop_size + 1)

        tasks = self.load_data(data_directory)
        device = next(self.parameters()).device
        children = [deepcopy(self) for _ in range(pop_size)]

        print('==> Evolving...')
        for epoch in range(epochs):
            print('==> Training children...')
            children = subset_gd(epoch, children)

            print('==> Selecting parent for next generation...')
            children = select(children)
            
        self.load_state_dict(children[0].state_dict())

    def calc_steps(self, steps: int, grid: th.FloatTensor, max_grid_area: int) -> int:
        '''
            For training by task, scale the allowed number of steps by the area of the grid.
        '''
        grid_area = grid.shape[0] * grid.shape[1]
        ratio = (grid_area / max_grid_area) ** 0.5
        scaled_steps = int(steps * ratio)
        scaled_steps = max(scaled_steps, steps // 4)
        return scaled_steps

    def fit_by_task(
        self, 
        task_path: str, 
        epochs: int = 800,
        steps: int = 10,
        trials: int = 128,
        learning_rate: float = 0.002, 
        mask_prob_low: float = 0.0, 
        mask_prob_high: float = 0.75, 
        force_sync: bool = False
    ) -> list[float]:
        '''
            Train NCA on one task.
        '''
        task = self.load_task(task_path)
        max_grid_area = max(len(example['input']) * len(example['input'][0]) for example in task['train'])
        device = next(self.parameters()).device

        optimizer = th.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0001 / learning_rate, total_iters=epochs)

        epoch_losses = []
        print('==> Training...')
        for epoch in range(epochs):
            random.shuffle(task['train'])
            
            batch_losses = []
            for example in task['train']:
                optimizer.zero_grad(set_to_none=True)
                x = th.tensor(example['input'], dtype=th.long, device=device)
                y = th.tensor(example['output'], dtype=th.long, device=device)

                steps_for_example = self.calc_steps(steps=steps, grid=x, max_grid_area=max_grid_area)

                losses = []
                for _ in range(trials):
                    states = self.rollout(state=x.unsqueeze(0), steps=steps_for_example, mask_prob_low=mask_prob_low, mask_prob_high=mask_prob_high, force_sync=force_sync)
                    _, loss = self.per_pixel_log_loss(states=states, target=y.unsqueeze(0))
                    losses.append(loss)

                avg_loss = th.stack(losses).mean()
                avg_loss.backward()
                batch_losses.append(avg_loss.item())
                optimizer.step()

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            progress_bar(epoch + 1, epochs, f"Loss: {epoch_loss:.4f}")
            scheduler.step()
        
        return epoch_losses

    @th.no_grad()
    def evaluate(self, inputs: th.LongTensor, targets: th.LongTensor, steps: int = 10) -> dict:
        '''
            Evaluate learned rules on new data.
        '''
        self.eval()
        device = next(self.parameters()).device
        states = self.rollout(state=inputs.to(device), steps=steps, mask_prob_low=0, mask_prob_high=0, force_sync=True)
        targets = targets.to(device)

        exact_match_accs = []
        pixel_accs = []
        for state in states:
            pred = self.decode(state)
            eq = (pred == targets).reshape(pred.shape[0], -1).all(dim=1)
            exact_match_accs.append(eq.float().mean().item())
            pixel_accs.append((pred == targets).float().mean().item())
        
        return {
            'exact_match_final_accuracy': exact_match_accs[-1],
            'exact_match_per_step_accuracies': exact_match_accs,
            'pixel_final_accuracy': pixel_accs[-1],
            'pixel_per_step_accuracies': pixel_accs,
            'final_state': self.decode(states[-1])
        }