import os
import argparse
import csv
from statistics import mean, median, pstdev
import re
from collections import defaultdict

def full_to_full_lexi(dataset: str, full_run: int, full_lexi_run: int, generations: int, epsilon: float, escheme: str):
    '''
        Records accuracy delta between the specified full model and the specified full lexi model.
    '''
    data = []
    fieldnames = ['task', 'accuracy_delta']

    with open(f'data/results/{dataset}_full/{dataset}_full_0{full_run}_results.csv', newline='', encoding='utf-8') as f: #! LEADING 0
        base_results = list(csv.DictReader(f))

    if escheme == 'mad':
        with open(f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{full_lexi_run}_({generations}g_MAD)_results.csv', newline='', encoding='utf-8') as f:
            lexi_results = list(csv.DictReader(f))
    elif escheme == 'bh':
        with open(f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{full_lexi_run}_({generations}g_BH)_results.csv', newline='', encoding='utf-8') as f:
            lexi_results = list(csv.DictReader(f))
    else:
        with open(f'data/results/{dataset}_full_lexi/{dataset}_full_lexi_{full_lexi_run}_({generations}g_{epsilon}e)_results.csv', newline='', encoding='utf-8') as f:
            lexi_results = list(csv.DictReader(f))
    
    for (base_row, lexi_row) in zip(base_results, lexi_results):
        data.append({
            'task': base_row['task'],
            'accuracy_delta': float(lexi_row['final_pixel_accuracy']) - float(base_row['final_pixel_accuracy'])
        })
    
    if escheme == 'mad':
        with open(f'data/results/{dataset}_comparisons/full_to_full_lexi/{full_run}_to_{full_lexi_run}_({generations}g_MAD).csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(data)
    elif escheme == 'bh':
        with open(f'data/results/{dataset}_comparisons/full_to_full_lexi/{full_run}_to_{full_lexi_run}_({generations}g_BH).csv', 'w', newline='', encoding='utf-8') as f:
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

def _parse_scheme(name: str, generations: int) -> str | None:
    m = re.search(rf"\(({generations})g_([^)]+)\)\.csv$", name)
    if not m:
        return None
    token = m.group(2)
    return token

def generate_table(dataset: str, full_run: int, full_lexi_run: int, generations: int):
    '''
        Generate table with comparison metrics for all models.
    '''
    full_to_full_lexi_dir = f'data/results/{dataset}_comparisons/full_to_full_lexi'
    by_scheme = defaultdict(list)

    for name in os.listdir(full_to_full_lexi_dir):
        if not name.startswith(f"{full_run}_to_{full_lexi_run}_") or f"({generations}g_" not in name:
            continue

        scheme = _parse_scheme(name, full_run, full_lexi_run, generations)

        if scheme is None:
            continue

        with open(os.path.join(full_to_full_lexi_dir, name), newline='', encoding='utf-8') as f:
            deltas = [float(r['accuracy_delta']) for r in csv.DictReader(f)]

        by_scheme[scheme] = deltas

    fieldnames = [
        'epsilon_scheme',
        'prop_tasks_improved',
        'avg_acc_delta',
        'median_acc_delta',
        'std',
    ]

    rows = []
    for scheme, deltas in sorted(by_scheme.items(), key=lambda kv: kv[0]):
        prop_improved = sum(1 for x in deltas if x > 0) / len(deltas)
        avg = mean(deltas)
        med = median(deltas)
        std = pstdev(deltas)

        rows.append({
            'epsilon_scheme': scheme,
            'prop_tasks_improved': f"{prop_improved:.4f}",
            'avg_acc_delta': f"{avg:.4f}",
            'median_acc_delta': f"{med:.4f}",
            'std': f"{std:.4f}",
        })

    with open(f"data/results/{dataset}_comparisons/{dataset}_run{full_lexi_run}_summary.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', type=str, help='Dataset on which the comparison will happen')

    base_to_lexi_parser = subparsers.add_parser('full_full_lexi', parents=[common])
    base_to_lexi_parser.add_argument('--fullrun', type=int, help='Run number of base model')
    base_to_lexi_parser.add_argument('--fulllexirun', type=int, help='Run number of lexi model')
    base_to_lexi_parser.add_argument('--gens', type=int, help='Number of generations lexi model was evolved')
    base_to_lexi_parser.add_argument('--epsilon', type=float, default=0, help='Survival threshold of lexi model')
    base_to_lexi_parser.add_argument('--escheme', type=str, help='Epsilon selection scheme')

    full_lexi_to_full_lexi_parser = subparsers.add_parser('full_lexi_full_lexi', parents=[common])
    full_lexi_to_full_lexi_parser.add_argument('--fullrun', type=int, help='Run number of base model')
    full_lexi_to_full_lexi_parser.add_argument('--fulllexirun', type=int, help='Run number of lexi model')
    full_lexi_to_full_lexi_parser.add_argument('--gens', type=int, help='Number of generations lexi model was evolved')
    full_lexi_to_full_lexi_parser.add_argument('--epsilons', nargs='+', type=float, help='Which values of epsilon the lexi models were trained with')

    table_parser = subparsers.add_parser('table', parents=[common])
    table_parser.add_argument('--fullrun', type=int, required=True, help='Run number of base model')
    table_parser.add_argument('--fulllexirun', type=int, required=True, help='Run number of lexi model')
    table_parser.add_argument('--gens', type=int, required=True, help='Number of generations lexi model was evolved')

    args = parser.parse_args()

    if args.command == 'full_full_lexi':
        full_to_full_lexi(
            dataset=args.dataset,
            full_run=args.fullrun,
            full_lexi_run=args.fulllexirun,
            generations=args.gens,
            epsilon=args.epsilon,
            escheme=args.escheme
        )
    elif args.command == 'full_lexi_full_lexi':
        full_lexi_to_full_lexi(
            dataset=args.dataset,
            full_run=args.fullrun,
            full_lexi_run=args.fulllexirun,
            generations=args.gens,
            epsilons=args.epsilons
        )
    elif args.command == 'table':
        generate_table(
            dataset=args.dataset,
            full_run=args.fullrun,
            full_lexi_run=args.fulllexirun,
            generations=args.gens
        )