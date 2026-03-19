from nca import NCA
import torch as th
import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--name', type=str, required=True, help='Model name')
    common.add_argument('--dataset', type=str, required=True, help='Dataset being trained on')
    common.add_argument('--nhidden', default=20, type=int, help='Number of hidden channels')
    common.add_argument('--temp', default=5, type=int, help='Temperature for softmaxing')
    common.add_argument('--epochs', default=800, type=int, help='Number of epochs')
    common.add_argument('--steps', default=10, type=int, help='Number of steps allowed')
    common.add_argument('--trials', default=128, type=int, help='Number of trials')
    common.add_argument('--lr', default=0.002, type=float, help='AdamW learning rate')
    common.add_argument('--mplow', default=0.0, type=float, help='Mask probability low')
    common.add_argument('--mphigh', default=0.75, type=float, help='Mask probability high')
    common.add_argument('--run', type=int, help='Run number')

    bytask = subparsers.add_parser('bytask', parents=[common], help='Train NCA on one task')
    bytask.add_argument('--task', type=int, required=True, help='Task being trained on')
    bytask.add_argument('--pop', default=4, type=int, help='Population size')
    bytask.add_argument('--epsilon', default=0, type=float, help='Survival threshold')
    bytask.add_argument('--lexi', action='store_true', help='Activate lexicase')
    bytask.add_argument('--casemode', required=True, type=str, help='Pixel scoring scheme')
    bytask.add_argument('--subfactor', type=int, default=2, help='Used to determine how many examples are sampled for subset_gd')
    bytask.add_argument('--escheme', type=str, required=True, help='Epsilon selection scheme')
    bytask.add_argument('--lrmax', default=0.1, type=float, help='Max learning rate for SGD (Lexi)')
    bytask.add_argument('--lrmin', default=0, type=float, help='Minimum learning rate for SGD (Lexi)')

    subparsers.add_parser('full', parents=[common], help='Train NCA on all tasks')

    full_lexi = subparsers.add_parser('full_lexi', parents=[common], help='Train NCA with gradient lexicase selection')
    full_lexi.add_argument('--pop', default=4, type=int, help='Population size')
    full_lexi.add_argument('--useavgloss', action='store_true', help='Use average loss to score children')
    full_lexi.add_argument('--epsilon', default=0, type=float, help='Survival threshold')
    full_lexi.add_argument('--casemode', required=True, type=str, help='What is used as test cases during lexicase selection')
    full_lexi.add_argument('--escheme', type=str, required=True, help='Epsilon selection scheme')
    full_lexi.add_argument('--lrmax', default=0.1, type=float, help='Max learning rate for SGD (Lexi)')
    full_lexi.add_argument('--lrmin', default=0, type=float, help='Minimum learning rate for SGD (Lexi)')
    full_lexi.add_argument('--test', action='store_true', help='Testing mode. Runs for 3 epochs.')

    args = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    print(f'==> DEVICE: {device}')
    model = NCA(n_hidden_channels=args.nhidden, temperature=args.temp)
    model = model.to(device)

    if args.command == 'bytask':
        if args.run is None:
            parser.error('--run is required for bytask checkpoints')

        losses = model.fit_by_task(
            task_path=f'../data/{args.dataset}/training/task_{args.task}.json',
            epsilon=args.epsilon,
            case_mode=args.casemode,
            lexi=args.lexi,
            epochs=args.epochs,
            epsilon_scheme=args.escheme,
            steps=args.steps,
            trials=args.trials,
            pop_size=args.pop,
            subset_factor=args.subfactor,
            adamw_learning_rate=args.lr,
            lr_max=args.lrmax,
            lr_min=args.lrmin,
            mask_prob_low=args.mplow,
            mask_prob_high=args.mphigh
        )

        '''
        with open(f'../data/losses/{args.dataset}_bytask/{args.run}/{args.name}_losses.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])
            for epoch, loss in enumerate(losses, start=1):
                writer.writerow([epoch, loss])
        '''
        if args.lexi:
            save_dir = f'../checkpoints/{args.dataset}_bytask_lexi/{args.run}/{args.name}_{args.escheme}_{args.casemode}.pth'
        else:
            save_dir = f'../checkpoints/{args.dataset}_bytask/{args.run}/{args.name}.pth'
        
        th.save({
            'model': model.state_dict(),
            'configs': {
                'n_hidden_channels': model.n_hidden_channels,
                'temperature': model.temperature,
                'steps': args.steps,
                'trials': args.trials,
                'learning_rate': args.lr,
                'mask_prob_low': args.mplow,
                'mask_prob_high': args.mphigh
            },
            'epochs': args.epochs,
            'device': str(device)
        }, save_dir)

    elif args.command == 'full':
        losses = model.fit(
            data_directory=f'../data/{args.dataset}/training',
            epochs=args.epochs,
            steps=args.steps,
            trials=args.trials,
            learning_rate=args.lr,
            mask_prob_low=args.mplow,
            mask_prob_high=args.mphigh
        )

        with open(f'../data/losses/{args.dataset}_full/{args.name}_losses.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])
            for epoch, loss in enumerate(losses, start=1):
                writer.writerow([epoch, loss])

        th.save({
            'model': model.state_dict(),
            'configs': {
                'n_hidden_channels': model.n_hidden_channels,
                'temperature': model.temperature,
                'steps': args.steps,
                'trials': args.trials,
                'learning_rate': args.lr,
                'mask_prob_low': args.mplow,
                'mask_prob_high': args.mphigh
            },
            'epochs': args.epochs,
            'device': str(device)
        }, f'../checkpoints/{args.dataset}_full/{args.name}.pth')

    elif args.command == 'full_lexi':
        losses = model.lexi_fit(
            data_directory=f'../data/{args.dataset}/training',
            epsilon=args.epsilon,
            case_mode=args.casemode,
            epsilon_scheme=args.escheme,
            epochs=args.epochs,
            steps=args.steps,
            trials=args.trials,
            lr_max=args.lrmax,
            lr_min=args.lrmin,
            mask_prob_low=args.mplow,
            mask_prob_high=args.mphigh,
            pop_size=args.pop,
            one_run_test=args.test
        )

        with open(f'../data/losses/{args.dataset}_full_lexi/{args.name}_losses.csv', 'w', newline='') as f:
            fieldnames = ['generation', 'child', 'loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(losses)

        epochs_for_ckpt = 3 if args.test else args.epochs * (args.pop + 1)

        save_dir = f'../checkpoints/{args.dataset}_full_lexi/{args.name}_({epochs_for_ckpt}g_{args.escheme}_{args.casemode}).pth'

        th.save({
            'model': model.state_dict(),
            'configs': {
                'n_hidden_channels': model.n_hidden_channels,
                'temperature': model.temperature,
                'pop_size': args.pop,
                'scored_with_avg': args.useavgloss,
                'steps': args.steps,
                'trials': args.trials,
                'learning_rate_max': args.lrmax,
                'learning_rate_min': args.lrmin,
                'mask_prob_low': args.mplow,
                'mask_prob_high': args.mphigh
            },
            'epochs': epochs_for_ckpt,
            'device': str(device)
        }, save_dir)

    print('==> Model saved.')


if __name__ == '__main__':
    main()