import os
from nca import NCA
import torch as th
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset being evaluated on')
    parser.add_argument('--run', type=int, help='Run number')
    parser.add_argument('--bytask', action='store_true', help='Was trained by task')
    args = parser.parse_args()

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    total = 0
    if args.bytask:
        num_tasks = len(os.listdir(f'../data/{args.dataset}/training'))

        for n in range(1, num_tasks + 1):
            ckpt = th.load(f'../checkpoints/{args.dataset}_bytask_0{args.run}/{args.dataset}_bytask{n}_0{args.run}.pth', map_location=th.device(device))
            state = ckpt['model']
            model = NCA()
            model.load_state_dict(state)

            with open(f'../data/{args.dataset}/training/task_{n}.json', 'r') as f:
                task = json.load(f)['test'][0]
            
            x = th.tensor(task['input'])
            y = th.tensor(task['output'])

            res = model.evaluate(inputs=x.unsqueeze(0), targets=y.unsqueeze(0))
            total += res['exact_match_final_accuracy']

            print(f"MODEL {n}: {res['exact_match_final_accuracy']}")

    print(f'TOTAL SOLVED: {total}')

if __name__ == '__main__':
    main()