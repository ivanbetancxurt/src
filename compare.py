import argparse
import csv

def full_to_full_lexi(dataset: str, full_run: int, full_lexi_run: int, generations: int, epsilon: float, mad: bool):
    '''
        Records accuracy delta between the specified full model and the specified full lexi model.
    '''
    data = []
    fieldnames = ['task', 'accuracy_delta']

    with open(f'data/results/{dataset}_full/{dataset}_full_0{full_run}_results.csv', newline='', encoding='utf-8') as f: #! LEADING 0
        base_results = list(csv.DictReader(f))

    if mad:
        with open(f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{full_lexi_run}_({generations}g_MAD)_results.csv', newline='', encoding='utf-8') as f:
            lexi_results = list(csv.DictReader(f))
    else:
        with open(f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{full_lexi_run}_({generations}g_{epsilon}e)_results.csv', newline='', encoding='utf-8') as f:
            lexi_results = list(csv.DictReader(f))
    
    for (base_row, lexi_row) in zip(base_results, lexi_results):
        data.append({
            'task': base_row['task'],
            'accuracy_delta': float(lexi_row['final_pixel_accuracy']) - float(base_row['final_pixel_accuracy'])
        })
    
    if mad:
        with open(f'data/results/{dataset}_comparisons/full_to_full_lexi/{full_run}_to_{full_lexi_run}_({generations}g_MAD).csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(data)
    else:
        with open(f'data/results/{dataset}_comparisons/full_to_full_lexi/{full_run}_to_{full_lexi_run}_({generations}g_{epsilon}e).csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(data)
    
def full_lexi_to_full_lexi(dataset: str, full_run: int, full_lexi_run: int, generations: int, epsilons: list[float]):
    '''
        Ranks the given full lexi models by number of positive accuracy deltas when compared to a full model (Exclusively for manual epsilon).
    '''

    scores = {}

    for epsilon in epsilons:
        with open(f'data/results/{dataset}_comparisons/full_to_full_lexi/{full_run}_to_{full_lexi_run}_({generations}g_{epsilon}e).csv', newline='', encoding='utf-8') as f:
            comparison = csv.DictReader(f)
            score = sum(1 for row in comparison if float(row['accuracy_delta']) >= 0)
            
        scores[epsilon] = score
    
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', type=str, help='Dataset on which the comparison will happen')

    base_to_lexi_parser = subparsers.add_parser('full_full_lexi', parents=[common])
    base_to_lexi_parser.add_argument('--fullrun', type=int, help='Run number of base model')
    base_to_lexi_parser.add_argument('--fulllexirun', type=int, help='Run number of lexi model')
    base_to_lexi_parser.add_argument('--gens', type=int, help='Number of generations lexi model was evolved')
    base_to_lexi_parser.add_argument('--epsilon', type=float, help='Survival threshold of lexi model')
    base_to_lexi_parser.add_argument('--mad', action='store_true', help='Lexi model used median absolute deviation for epsilon')

    full_lexi_to_full_lexi_parser = subparsers.add_parser('full_lexi_full_lexi', parents=[common])
    full_lexi_to_full_lexi_parser.add_argument('--fullrun', type=int, help='Run number of base model')
    full_lexi_to_full_lexi_parser.add_argument('--fulllexirun', type=int, help='Run number of lexi model')
    full_lexi_to_full_lexi_parser.add_argument('--gens', type=int, help='Number of generations lexi model was evolved')
    full_lexi_to_full_lexi_parser.add_argument('--epsilons', nargs='+', type=float, help='Which values of epsilon the lexi models were trained with')

    args = parser.parse_args()

    if args.command == 'full_full_lexi':
        full_to_full_lexi(
            dataset=args.dataset,
            full_run=args.fullrun,
            full_lexi_run=args.fulllexirun,
            generations=args.gens,
            epsilon=args.epsilon,
            mad=args.mad
        )
    elif args.command == 'full_lexi_full_lexi':
        full_lexi_to_full_lexi(
            dataset=args.dataset,
            full_run=args.fullrun,
            full_lexi_run=args.fulllexirun,
            generations=args.gens,
            epsilons=args.epsilons
        )