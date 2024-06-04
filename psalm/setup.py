from setuptools import setup, find_packages

setup(
    name='protein-sequence-annotation',
    version='0.2.4',
    description='Protein sequence annotation with language models',
    url='https://github.com/Protein-Sequence-Annotation/PSALM',
    packages=find_packages(),
    install_requires=[
        'huggingface-hub>=0.23.2',
        'safetensors>=0.4.3',
        'fair-esm>=2.0.0'
    ],
    author='Arpan Sarkar & Kumaresh Krishnan',
    author_email='arpan_sarkar@g.harvard.edu',
    license='CC by 4.0',
    python_requires ='>=3.10'
)