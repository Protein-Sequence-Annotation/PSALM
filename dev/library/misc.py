from Bio import SeqIO
import pickle

ref_file = f'../data/processing/uniref50.fasta'
test_file = f'../data/test_fasta/test_ids_full.fasta'

ref_ids = set()
test_ids = set()

ctr = 0

for record in SeqIO.parse(ref_file, 'fasta'):

    ref_ids.add(record.id)
    ctr += 1

    if ctr == 100:
        break

for record in SeqIO.parse(test_file, 'fasta'):

    test_ids.add(record.id)

additional = test_ids.difference(ref_ids)

with open(f'../data/uniref_unique.pkl', 'wb') as f:
    pickle.dump({'unique': additional}, f)
