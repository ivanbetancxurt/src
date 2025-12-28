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
    parser.add_argument('--lexi', action='store_true', help='Was evolved with lexi')
    parser.add_argument('--gens', type=int, help='Number of generations')
    parser.add_argument('--epsilon', type=float, help='Survival threshold')
    args = parser.parse_args()

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    data = []
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
    lexi_fieldnames = [
        'task', 
        'solved', 
        'final_pixel_accuracy', 
        'n_hidden_channels',
        'temperature',
        'steps',
        'trials',
        'learning_rate_max',
        'learning_rate_min',
        'mask_prob_low',
        'mask_prob_high'
    ]

    def record():
        if args.bytask:
            with open(f'../data/results/{args.dataset}_bytask/{args.dataset}_bytask_0{args.run}_results.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()  
                writer.writerows(data)
        else:
            with open(f'../data/results/{args.dataset}_full/{args.dataset}_full_0{args.run}_results.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()  
                writer.writerows(data)

        if args.lexi:
            with open(f'../data/results/{args.dataset}_full/{args.dataset}_full_lexi_{args.run}_results.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=lexi_fieldnames)
                writer.writeheader()  
                writer.writerows(data)


    def evaluate(model: NCA, configs: dict, task_num: int, dataset: str):
        '''
            Evaluate the model on the specified task.
        '''
        with open(f'../data/{dataset}/training/task_{task_num}.json', 'r') as f:
            task = json.load(f)['test'][0]
            
        x = th.tensor(task['input'])
        y = th.tensor(task['output'])

        res = model.evaluate(inputs=x.unsqueeze(0), targets=y.unsqueeze(0))

        if args.lexi:
            data.append({
                'task': task_num,
                'solved': 'True' if res['exact_match_final_accuracy'] == 1.0 else 'False',
                'final_pixel_accuracy': res['pixel_final_accuracy'],
                'n_hidden_channels': configs['n_hidden_channels'],
                'temperature': configs['temperature'],
                'steps': configs['steps'],
                'trials': configs['trials'],
                'learning_rate_max': configs['learning_rate_max'],
                'learning_rate_min': configs['learning_rate_min'],
                'mask_prob_low': configs['mask_prob_low'],
                'mask_prob_high': configs['mask_prob_high']
            })
        else:
            data.append({
                'task': task_num,
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

    num_tasks = len(os.listdir(f'../data/{args.dataset}/training'))
    if args.bytask:
        for n in range(1, num_tasks + 1):
            ckpt = th.load(f'../checkpoints/{args.dataset}_bytask/0{args.run}/{args.dataset}_bytask{n}_0{args.run}.pth', map_location=th.device(device))
            configs = ckpt['configs']
            state = ckpt['model']
            model = NCA()
            model.load_state_dict(state)
            model.to(device)

            evaluate(model=model, configs=configs, task_num=n, dataset=args.dataset)

        record()
    else:
        ckpt = th.load(f'../checkpoints/{args.dataset}_full/{args.dataset}_full_0{args.run}.pth', map_location=th.device(device))
        configs = ckpt['configs']
        state = ckpt['model']
        model = NCA()
        model.load_state_dict(state)
        model.to(device)

        for n in range(1, num_tasks + 1):
            evaluate(model=model, configs=configs, task_num=n, dataset=args.dataset)

        record()
        
    if args.lexi:
        ckpt = th.load(f'../checkpoints/{args.dataset}_full_lexi/{args.dataset}_full_lexi_{args.run}_({args.gens}g_{args.epsilon}e).pth', map_location=th.device(device))
        configs = ckpt['configs']
        state = ckpt['model']
        model = NCA()
        model.load_state_dict(state)
        model.to(device)

        for n in range(1, num_tasks + 1):
            evaluate(model=model, configs=configs, task_num=n, dataset=args.dataset)
        
        record()

if __name__ == '__main__':
    main()