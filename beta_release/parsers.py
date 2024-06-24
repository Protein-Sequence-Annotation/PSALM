import argparse

def train_parser():

    parser = argparse.ArgumentParser(prog='PSALM_train', description='Trains the PSALM Model')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-m', '--model', type=str, default='fam', choices=['fam', 'clan'], help='model name - fam/clan')
    parser.add_argument('-r', '--resume', default='none', type=str, help='model path to resume training')
    parser.add_argument('-ne', '--num-epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('-ns', '--num-shards', type=int, default=50, help='number of shards')
    parser.add_argument('-o', '--output', type=str, help='folder name for results', required=True)
    parser.add_argument('-s', '--no-seed', action='store_false', default=True, help='set seed for reproducibility')
    parser.add_argument('-l', '--no-log', action='store_false', default=True, help='log to wandb')
    parser.add_argument('-rt', '--root', type=str, default='datasets', help='path to data folder')
    parser.add_argument('-x', '--suffix', type=str, default='', help='suffix to use for train dataset')
    parser.add_argument('-v', '--no-validation', action='store_false', default=True, help='compute validation loss')
    parser.add_argument('-p', '--project', type=str, default='PSALM-1b', help='wandb project name')

    return parser

def test_parser():

    parser = argparse.ArgumentParser(prog='PSALM_test', description='Tests the PSALM Model')
    parser.add_argument('-m', '--model', type=str, default='fam', choices=['fam', 'clan'], help='model name - fam/clan')
    parser.add_argument('-ns', '--num-shards', type=int, default=50, help='number of shards')
    parser.add_argument('-ff', '--fam-filename', type=str, default='models/fam.pth', help='path to family classifier model')
    parser.add_argument('-cf', '--clan-filename', type=str, default='models/clan.pth', help='path to clan classifier model')
    parser.add_argument('-s', '--no-seed', action='store_false', default=True, help='set seed for reproducibility')
    parser.add_argument('-rt', '--root', type=str, default='datasets', help='path to data folder')
    parser.add_argument('-x', '--suffix', type=str, default='', help='suffix to use for test dataset')

    return parser

def single_parser():

    parser = argparse.ArgumentParser(prog='PSALM_test', description='Tests the PSALM Model')
    parser.add_argument('-ns', '--num-shards', type=int, default=50, help='number of shards')
    parser.add_argument('-i', '--input', type=str, default='none', help='filepath to custom input fasta file')
    parser.add_argument('-ff', '--fam-filename', type=str, default='models/fam.pth', help='path to family classifier model')
    parser.add_argument('-cf', '--clan-filename', type=str, default='models/clan.pth', help='path to clan classifier model')
    parser.add_argument('-s', '--no-seed', action='store_false', default=True, help='set seed for reproducibility')
    parser.add_argument('-rt', '--root', type=str, default='datasets', help='path to data folder')
    parser.add_argument('-t', '--thresh', type=float, default=0.72, help='minimum assignment probability needed for annotation')
    parser.add_argument('-x', '--suffix', type=str, default='', help='suffix to use for test dataset')

    return parser