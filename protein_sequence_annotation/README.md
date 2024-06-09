# PSALM
This package contains code and pre-trained weights for Protein Sequence Annotation with Language Models (PSALM) from our [2024 preprint](https://www.biorxiv.org/content/10.1101/2024.06.04.596712v1).

## Abstract
Protein function inference relies on annotating protein domains via sequence similarity, often modeled through profile Hidden Markov Models (profile HMMs), which capture evolutionary diversity within related domains. However, profile HMMs make strong simplifying independence assumptions when modeling residues in a sequence. Here, we introduce PSALM (Protein Sequence Annotation with Language Models), a hierarchical approach that relaxes these assumptions and uses representations of protein sequences learned by protein language models to enable high-sensitivity, high-specificity residue-level protein sequence annotation. We validate PSALM's performance on a curated set of "ground truth" annotations determined by a profile HMM-based method and highlight PSALM as a promising alternative for protein sequence annotation.

## Usage
PSALM requires Python>=3.10 and PyTorch>=2.2.0. Start a fresh conda environment to use PSALM:

```
conda create -n "psalm" python=3.10
conda activate psalm
pip install torch protein-sequence-annotation notebook ipykernel
python -m ipykernel install --user
```

OR just install PSALM alone by using the [`protein-sequence-annotation` PyPI package](https://pypi.org/project/protein-sequence-annotation/#description). 
```
pip install protein-sequence-annotation
```

After the pip install, you can load and use a pretrained model as follows:
```python
import torch
from psalm import psalm

# Load PSALM clan and fam models
PSALM = psalm(clan_model_name="ProteinSequenceAnnotation/PSALM-1-clan",
             fam_model_name="ProteinSequenceAnnotation/PSALM-1-family",
             device = 'cpu') #cpu by default, replace with 'cuda' or 'mps' as needed

# Prepare data (use PSALM.read_fasta(fasta_file_path) to get data directly from a FASTA file)
data = [
    ("Human Beta Globin", "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"),
    ("Flavohemoprotein", "MLDAQTIATVKATIPLLVETGPKLTAHFYDRMFTHNPELKEIFNMSNQRNGDQREALFNAIAAYASNIENLPALLPAVEKIAQKHTSFQIKPEQYNIVGEHLLATLDEMFSPGQEVLDAWGKAYGVLANVFINREAEIYNENASKAGGWEGTRDFRIVAKTPRSALITSFELEPVDGGAVAEYRPGQYLGVWLKPEGFPHQEIRQYSLTRKPDGKGYRIAVKREEGGQVSNWLHNHANVGDVVKLVAPAGDFFMAVADDTPVTLISAGVGQTPMLAMLDTLAKAGHTAQVNWFHAAENGDVHAFADEVKELGQSLPRFTAHTWYRQPSEADRAKGQFDSEGLMDLSKLEGAFSDPTMQFYLCGPVGFMQFTAKQLVDLGVKQENIHYECFGPHKVL")
]

# Visualize PSALM annotations (add optional save_path argument: PSALM.annotate(data,save_path="save_folder")
PSALM.annotate(data)
```

## Cite
If you find PSALM useful in your research, please cite the following paper:
```bibtex
@article {sarkarkrishnan2024psalm,
	author = {Sarkar, Arpan and Krishnan, Kumaresh and Eddy, Sean R},
	title = {Protein Sequence Domain Annotation using Language Models},
	year = {2024},
	URL = {https://www.biorxiv.org/content/early/2024/06/05/2024.06.04.596712},
	journal = {bioRxiv}
}

```
