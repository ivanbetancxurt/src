import os
from nca import NCA
import torch as th
import json
import argparse
import csv

def main():
    #* EXAMPLE COMMAND: python3 evaluate.py full --dataset "arc1" --run 2 single --task 153

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', type=str, required=True, help='Dataset being evaluated on')
    common.add_argument('--run', type=int, required=True, help='Run number')

    subparsers.add_parser('bytask', parents=[common], help='Trained NCA on one tasks')

    full = subparsers.add_parser('full', parents=[common], help='Trained NCA on all tasks')
    full_sub = full.add_subparsers(dest='variant')
    full_single = full_sub.add_parser('single', help='Evaluate a single task')
    full_single.add_argument('--task', type=int, required=True, help='Task being evaluated on')

    full_lexi = subparsers.add_parser('full_lexi', parents=[common], help='Trained NCA on all tasks with lexi')
    full_lexi_sub = full_lexi.add_subparsers(dest='variant')
    full_lexi.add_argument('--gens', type=int, help='Number of generations')
    full_lexi.add_argument('--epsilon', type=float, help='Survival threshold')
    full_lexi.add_argument('--mad', action='store_true', help='Used median absolute deviation for epsilon')
    full_lexi_single = full_lexi_sub.add_parser('single', help='Evaluate a single task')
    full_lexi_single.add_argument('--task', type=int, required=True)
    
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
        'scored_with_avg',
        'pop_size', 
        'n_hidden_channels',
        'temperature',
        'steps',
        'trials',
        'learning_rate_max',
        'learning_rate_min',
        'mask_prob_low',
        'mask_prob_high'
    ]

    def record(command: str):
        if command == 'bytask' or command == 'full':
            with open(f'../data/results/{args.dataset}_{command}/{args.dataset}_{command}_{args.run}_results.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()  
                    writer.writerows(data)
        else:
            if args.mad:
                with open(f'../data/results/{args.dataset}_{command}/{args.dataset}_{command}_{args.run}_({args.gens}g_MAD)_results.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=lexi_fieldnames)
                    writer.writeheader()  
                    writer.writerows(data)
            else:
                with open(f'../data/results/{args.dataset}_{command}/{args.dataset}_{command}_{args.run}_({args.gens}g_{args.epsilon}e)_results.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=lexi_fieldnames)
                    writer.writeheader()  
                    writer.writerows(data)
            

    def evaluate(model: NCA, configs: dict, task_num: int, dataset: str, generate_img: bool = False):
        '''
            Evaluate the model on the specified task.
        '''
        with open(f'../data/{dataset}/training/task_{task_num}.json', 'r') as f:
            task = json.load(f)['test'][0]
            
        x = th.tensor(task['input'])
        y = th.tensor(task['output'])

        res = model.evaluate(inputs=x.unsqueeze(0), targets=y.unsqueeze(0), generate_img=generate_img)

        if args.command == 'full_lexi':
            data.append({
                'task': task_num,
                'solved': 'True' if res['exact_match_final_accuracy'] == 1.0 else 'False',
                'final_pixel_accuracy': res['pixel_final_accuracy'],
                'scored_with_avg': configs['scored_with_avg'],
                'pop_size': configs['pop_size'],
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
    model = NCA()

    if args.command == 'bytask':
        for n in range(1, num_tasks + 1):
            ckpt = th.load(f'../checkpoints/{args.dataset}_bytask/{args.run}/{args.dataset}_bytask{n}_{args.run}.pth', map_location=th.device(device))
            configs = ckpt['configs']
            state = ckpt['model']
            model.load_state_dict(state)
            model.to(device)

            evaluate(model=model, configs=configs, task_num=n, dataset=args.dataset)

        record(args.command)
    elif args.command == 'full':
        ckpt = th.load(f'../checkpoints/{args.dataset}_full/{args.dataset}_full_{args.run}.pth', map_location=th.device(device))
        state = ckpt['model']
        configs = ckpt['configs']
        model.load_state_dict(state)
        model.to(device)

        if args.variant == 'single':
            evaluate(model=model, configs=configs, task_num=args.task, dataset=args.dataset, generate_img=True)

        else:
            for n in range(1, num_tasks + 1):
                evaluate(model=model, configs=configs, task_num=n, dataset=args.dataset)

            record(args.command)
    elif args.command == 'full_lexi':
        if args.mad:
            ckpt = th.load(f'../checkpoints/{args.dataset}_full_lexi/{args.dataset}_full_lexi_{args.run}_({args.gens}g_MAD).pth', map_location=th.device(device))
        else:
            ckpt = th.load(f'../checkpoints/{args.dataset}_full_lexi/{args.dataset}_full_lexi_{args.run}_({args.gens}g_{args.epsilon}e).pth', map_location=th.device(device))
        configs = ckpt['configs']
        state = ckpt['model']
        model.load_state_dict(state)
        model.to(device)

        for n in range(1, num_tasks + 1):
            evaluate(model=model, configs=configs, task_num=n, dataset=args.dataset)
        
        record(args.command)


if __name__ == '__main__':
    main()