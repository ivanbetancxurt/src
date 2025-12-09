import os
from nca import NCA
import torch as th
import json
import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset being evaluated on')
    parser.add_argument('--run', type=int, help='Run number')
    parser.add_argument('--bytask', action='store_true', help='Was trained by task')
    args = parser.parse_args()

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    data = []
    if args.bytask:
        num_tasks = len(os.listdir(f'../data/{args.dataset}/training'))

        for n in range(1, num_tasks + 1):
            ckpt = th.load(f'../checkpoints/{args.dataset}_bytask_0{args.run}/{args.dataset}_bytask{n}_0{args.run}.pth', map_location=th.device(device))
            configs = ckpt['configs']
            state = ckpt['model']
            model = NCA()
            model.load_state_dict(state)

            with open(f'../data/{args.dataset}/training/task_{n}.json', 'r') as f:
                task = json.load(f)['test'][0]
            
            x = th.tensor(task['input'])
            y = th.tensor(task['output'])

            res = model.evaluate(inputs=x.unsqueeze(0), targets=y.unsqueeze(0))

            fieldnames = [
                'task', 
                'solved', 
                'final_pixel_accuracy', 
                'n_hidden_channels',
                'temperature',
                'steps',
                'trials',
                'learning_rate',
                'mask_prob_low',
                'mask_prob_high'
            ]

            data.append({
                'task': n,
                'solved': 'True' if res['exact_match_final_accuracy'] == 1.0 else 'False',
                'final_pixel_accuracy': res['pixel_final_accuracy'],
                'n_hidden_channels': configs['n_hidden_channels'],
                'temperature': configs['temperature'],
                'steps': configs['steps'],
                'trials': configs['trials'],
                'learning_rate': configs['learning_rate'],
                'mask_prob_low': configs['mask_prob_low'],
                'mask_prob_high': configs['mask_prob_high']
            })

        with open(f'../data/results/results_{args.dataset}_bytask_0{args.run}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(data)


if __name__ == '__main__':
    main()