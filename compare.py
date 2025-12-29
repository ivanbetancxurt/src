import argparse

def base_to_lexi():
    fieldnames = ['task', 'accuracy_delta']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    base_to_lexi_parser = subparsers.add_parser('baselexi')
    base_to_lexi_parser.add_argument('--dataset', type=str, help='Dataset on which the comparison will happen')
    base_to_lexi_parser.add_argument('--baserun', type=int, help='Run number of base model')
    base_to_lexi_parser.add_argument('--lexirun', type=int, help='Run number of lexi model')
    base_to_lexi_parser.add_argument('--gens', type=int, help='Number of generations lexi model was evolved')
    base_to_lexi_parser.add_argument('--epsilon', type=float, help='Survival threshold of lexi model')
