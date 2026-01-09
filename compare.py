import argparse
import csv

def base_to_lexi(dataset: str, base_run: int, lexi_run: int, generations: int, epsilon: float, mad: bool):
    data = []
    fieldnames = ['task', 'accuracy_delta']

    with open(f'data/results/{dataset}_full/{dataset}_full_0{base_run}_results.csv', newline="", encoding="utf-8") as f: #! LEADING 0
        base_results = list(csv.DictReader(f))

    if mad:
        with open(
            f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{lexi_run}_({generations}g_MAD)_results.csv', 
            newline="", 
            encoding="utf-8"
        ) as f:
            lexi_results = list(csv.DictReader(f))
    else:
        with open(
            f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{lexi_run}_({generations}g_{epsilon}e)_results.csv', 
            newline="", 
            encoding="utf-8"
        ) as f:
            lexi_results = list(csv.DictReader(f))
    
    for (base_row, lexi_row) in zip(base_results, lexi_results):
        data.append({
            'task': base_row['task'],
            'accuracy_delta': float(lexi_row['final_pixel_accuracy']) - float(base_row['final_pixel_accuracy'])
        })
    
    if mad:
        with open(f'data/results/comparisons/base_to_lexi/{base_run}_to_{lexi_run}_({generations}g_MAD).csv', 'w', newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(data)
    else:
        with open(f'data/results/comparisons/base_to_lexi/{base_run}_to_{lexi_run}_({generations}g_{epsilon}e).csv', 'w', newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    base_to_lexi_parser = subparsers.add_parser('baselexi')
    base_to_lexi_parser.add_argument('--dataset', type=str, help='Dataset on which the comparison will happen')
    base_to_lexi_parser.add_argument('--baserun', type=int, help='Run number of base model')
    base_to_lexi_parser.add_argument('--lexirun', type=int, help='Run number of lexi model')
    base_to_lexi_parser.add_argument('--gens', type=int, help='Number of generations lexi model was evolved')
    base_to_lexi_parser.add_argument('--epsilon', type=float, help='Survival threshold of lexi model')
    base_to_lexi_parser.add_argument('--mad', action='store_true', help='Lexi model used median absolute deviation for epsilon')

    args = parser.parse_args()

    if args.command == 'baselexi':
        base_to_lexi(
            dataset=args.dataset,
            base_run=args.baserun,
            lexi_run=args.lexirun,
            generations=args.gens,
            epsilon=args.epsilon,
            mad=args.mad
        )
