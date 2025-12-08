from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='protein_sequence_annotation',
    version='1.1.3',
    description='Protein sequence annotation with language models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Protein-Sequence-Annotation/PSALM',
    packages=find_packages(),
    install_requires=[
        'huggingface-hub>=0.23.2',
        'safetensors>=0.4.3',
        'fair-esm>=2.0.0',
        'matplotlib>=3.8.2',
        'biopython>=1.83'
    ],
    author='Arpan Sarkar & Kumaresh Krishnan',
    author_email='arpan_sarkar@g.harvard.edu',
    license='CC by 4.0',
    python_requires ='>=3.10'
)