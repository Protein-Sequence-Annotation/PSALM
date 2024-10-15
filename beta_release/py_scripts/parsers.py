import argparse

def train_parser():

    parser = argparse.ArgumentParser(prog='PSALM_train', description='Trains the PSALM Model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-ln', '--layer_number', type=int, default=33, help='ESM2 layer from which to get the embeddings')
    parser.add_argument('-w', '--warmup_percentage', type=int, default=0, help='percent of epoch to warmup')
    parser.add_argument('-m', '--mode', type=str, default='fam', choices=['fam', 'clan','eval', 'only'], help='model name - fam/clan or evaluate full pipeline')
    parser.add_argument('-r', '--resume', default='none', type=str, help='model path to resume training')
    parser.add_argument('-ne', '--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('-o', '--output', type=str, help='folder name for results', required=True)
    parser.add_argument('-ns', '--no_seed', action='store_true', default=False, help='set seed for reproducibility') ### CHECK SEED
    parser.add_argument('-nl', '--no_log', action='store_true', default=False, help='log to wandb')
    parser.add_argument('-rt', '--root', type=str, default='datasets', help='path to data folder')
    parser.add_argument('-x', '--suffix', type=str, default='', help='suffix to use for train dataset')
    parser.add_argument('-nv', '--no_validation', action='store_true', default=False, help='compute validation loss')
    parser.add_argument('-p', '--project', type=str, default='PSALM-revised', help='wandb project name')
    parser.add_argument('-ff', '--fam-filename', type=str, default='models/fam.pth', help='path to family classifier model')
    parser.add_argument('-cf', '--clan-filename', type=str, default='models/clan.pth', help='path to clan classifier model')
    parser.add_argument('-es', '--esm-size', type=str, default='t33_650M', help='ESM Model Size')

    return parser

def test_parser():

    parser = argparse.ArgumentParser(prog='PSALM_test', description='Tests the PSALM Model')
    parser.add_argument('-m', '--model', type=str, default='fam', choices=['fam', 'clan'], help='model name - fam/clan')
    parser.add_argument('-ff', '--fam-filename', type=str, default='models/fam.pth', help='path to family classifier model')
    parser.add_argument('-cf', '--clan-filename', type=str, default='models/clan.pth', help='path to clan classifier model')
    parser.add_argument('-s', '--no-seed', action='store_false', default=True, help='set seed for reproducibility')
    parser.add_argument('-rt', '--root', type=str, default='datasets', help='path to data folder')
    parser.add_argument('-x', '--suffix', type=str, default='_20', help='suffix to use for test dataset')

    return parser

def single_parser():

    parser = argparse.ArgumentParser(prog='PSALM_test', description='Tests the PSALM Model')
    parser.add_argument('-i', '--input', type=str, default='none', help='filepath to custom input fasta file')
    parser.add_argument('-ff', '--fam-filename', type=str, default='models/fam.pth', help='path to family classifier model')
    parser.add_argument('-cf', '--clan-filename', type=str, default='models/clan.pth', help='path to clan classifier model')
    parser.add_argument('-s', '--no-seed', action='store_false', default=True, help='set seed for reproducibility')
    parser.add_argument('-rt', '--root', type=str, default='datasets', help='path to data folder')
    parser.add_argument('-t', '--thresh', type=float, default=0.72, help='minimum assignment probability needed for annotation')
    parser.add_argument('-x', '--suffix', type=str, default='', help='suffix to use for test dataset')

    return parser