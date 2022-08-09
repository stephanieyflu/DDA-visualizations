# SicknessMiner

## Citation

SicknessMiner: a deep-learning-driven text-mining tool to abridge disease-disease associations
Nícia Rosário-Ferreira 1,2,⤉*, Victor Guimarães 3,4,⤉, Vitor S. Costa 3,4, Irina S. Moreira 5,6 - 2021, Submitted.

## Installation

In order to install this system, the user needs two different python 
environments: a python 3.5 environment for NormCo, and a Python 3.7 
environment for SicknessMiner.

We suggest the use of [Anaconda3](https://www.anaconda.com) in order to 
create both environments without changing a possible existing environment in 
the host machine.

In the following instructions, we assume the user has already cloned this 
repository and it is in this repository directory.

### NormCo Installation

Install [NormCo](https://github.com/IBM/aihn-ucsd) following the instruction
in their github page.

We also listed the instructions to install NormCo herein, in order to make this 
README self-contained.

#### Steps

```
# Moving to parent directory
cd ..

# Cloning NormCo repository
git clone https://github.com/IBM/aihn-ucsd.git

# Moving to NormCo directory
cd aihn-ucsd/NormCo-deep-disease-normalization

# Creating and activating conda environment
conda create -n entity_normalization python=3.5
source ~/anaconda3/bin/activate
. activate entity_normalization

# Installing NormCo dependences
conda install --yes --file requirements.txt
conda install pytorch==0.4.0 torchvision -c pytorch

# Installing additional dependences
conda install --yes pandas tqdm nltk scikit-learn spacy smart_open

# Downloading nltk auxilary files
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

# Check the path of the python command of the environment
# Write down the result of this command, it might be useful later
which python

# Deactivating NormCo environment
conda deactivate
```

After installing NormCo, we should place it in the SicknessMiner directory. 

```
# Moving to SicknessMiner parent directory
cd ../..

# Moving NormCo to SicknessMiner directory
mv aihn-ucsd/NormCo-deep-disease-normalization SicknessMiner

# Removing the remaining files
rm -rf aihn-ucsd
```

### SicknessMiner Installation

Now that we have NormCo, we can install SicknessMiner.

#### Steps

```
# Entering SicknessMiner directory
cd SicknessMiner

# Creating and activating conda environment
conda create -n venv python=3.7
. activate venv

# Installing SicknessMiner dependences
conda install --yes --file requirements.txt

# Installing bert-for-tf2
git clone https://github.com/kpe/bert-for-tf2.git

cd bert-for-tf2

python setup.py install

cd ..

# Removing bert-for-tf2 files
rm -rf bert-for-tf2

# Deactivating NormCo environment
# Skip this if you are going to run SicknessMiner next
conda deactivate
```

#### Download the Pre-trained Models

In order to use SicknessMiner, one must download the pre-trained models 
from [here](https://figshare.com/s/04259fac69da301680c2).

In order to do so, follow the steps below.

```
# Creating the model's directory
mkdir models

# Downloading the NER model
wget https://ndownloader.figshare.com/files/28278432?private_link=04259fac69da301680c2 -O NER_SicknessMiner.zip
unzip NER_SicknessMiner.zip
rm NER_SicknessMiner.zip

# Downloading the NEN model
wget https://ndownloader.figshare.com/files/28278168?private_link=04259fac69da301680c2 -O NEN_SicknessMiner.zip
unzip NEN_SicknessMiner.zip
rm NEN_SicknessMiner.zip
```

The NER model was derived from the BioBERT models, which was a courtesy of the 
U.S. National Library of Medicine and can be found at 
https://github.com/dmis-lab/biobert

## Run

The steps bellow are used to run SicknessMiner. We assume the user is in 
the root directory of this repository.

The input files to SicknessMiner should be in 
[PubTator](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/Format.html)
format, ad there must be, at least, a blank line between the articles.

```
# Activating conda environment
# Skip this if you already have the environment activated
. activate venv

# Config environment variables
export PYTHONPATH=$(pwd)
export CONDA_ENV=<PATH_TO_NORMCO_PYTHON_ENV>

# Run SicknessMiner
python sicknessminer/main.py <INPUT_FILE> <OUTPUT_FILE>
```

The CONDA_ENV variable must point to the python of the NormCo environment. 
That is usually placed inside anaconda directory.

One can check this by using the command `which python` with the NormCo 
environment activated.

The example below includes a possible valid path:

```
export CONDA_ENV=~/anaconda3/envs/entity_normalization/bin/python
export CONDA_ENV=~/anaconda3/envs/entity_normalization/python
```

### Example

Assuming all went well, you can now run the example:

```
python sicknessminer/main.py example/input.txt example/output.txt
```

The `example/output.txt` file should be equal to the `example/expected.txt` 
file.

The `example/input.txt` contains the first abstract from the train file 
obtained from the 
[NCBI Disease dataset](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/). 

If one wants to use raw text files as input, he/she can do so by passing the 
--raw/-r alongside the input arguments, as shown in example below: 

```
python sicknessminer/main.py --raw example/input_raw.txt example/output_raw.txt
```

or

```
python sicknessminer/main.py -r example/input_raw.txt example/output_raw.txt
```

The `example/output_raw.txt` file should be equal to the 
`example/expected_raw.txt` file.

## Clean

Unsuccessful runs of the system might create temporary files in the `tmp` 
directory. These files are kept their for debugging propose, but they can take 
too much space.

In order to clean those files, use the command:
```
rm -rf tmp/
```

## Results

The directory `results` contains the results from the analysis presented in our 
paper mentioned above.

The files `SicknessMiner.tsv` and `DisGeNET.tsv` shows the diseases related
to the target Blood Cancer diseases (BCs). The diseases in the
`SicknessMiner.tsv` file are sorted by the number of co-mentions between them
and the target disease, in a descending order. Whereas the diseases in the 
`DisGeNET.tsv` file are sorted by the Jaccard index between the set of genes 
related to the target disease and the set of genes related to the associated 
disease, in a descending order.

The meaning of the columns are described below.

For the `SicknessMiner.tsv`:
1. Index_disease: the name of the target BC disease;
2. Index_disease_id: the MeSH code of the target BC disease;
3. Associated_disease: the name of the associated disease;
4. Associated_disease_id: the MeSH code of the associated disease;
5. Co-mentions: the number of co-mentions between the target and the 
   associated disease.

For the `DisGeNET.tsv`:
1. Index_disease: the name of the target BC disease;
2. Index_disease_id: the UMLS CUI code of the target BC disease;
3. Associated_disease: the name of the associated disease;
4. Associated_disease_id: the UMLS CUI code of the associated disease;
5. Jaccard_index: the Jaccard index between the set of genes related to the 
   target disease and the set of genes related to the associated disease;
6. MESH_id: the MeSH id of the associated disease, if known;
7. OMIN_id: the OMIN id of the associated disease, if known.

## Additional Notes by Stephanie

Input format:
- Place all abstracts in the `input_files` folder as individual .txt files
- The title of the abstract can also be included in the .txt file

## Resources

SicknessMiner has two main models, the NER which uses a BioBERT model and 
NEN, which uses NormCo. We reimplemented the BioBERT model using the library 
bert-for-tf2. In addition, some pieces of code were obtained from the 
[standoff2conll](https://github.com/spyysalo/standoff2conll) project.

If you use this system, please cite the following resources:

### [BioBERT](https://github.com/dmis-lab/biobert)
```
@article{10.1093/bioinformatics/btz682,
    title = "{BioBERT: a pre-trained biomedical language representation model for biomedical text mining}",
    author = {Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
    journal = {Bioinformatics},
    year = {2019},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz682},
    url = {https://doi.org/10.1093/bioinformatics/btz682},
}
```

#### [BERT](https://github.com/google-research/bert)
```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

### [NormCo](https://github.com/IBM/aihn-ucsd/tree/master/NormCo-deep-disease-normalization)
```
@inproceedings {wright2019normco,
    title={NormCo: Deep Disease Normalization for Biomedical Knowledge Base Construction},
    author={Wright, Dustin and Katsis, Yannis and Mehta, Raghav and Hsu, Chun-Nan},
    booktitle={Automated Knowledge Base Construction},
    year={2019},
    url={https://openreview.net/forum?id=BJerQWcp6Q},
}
```

### [bert-for-tf2](https://github.com/kpe/bert-for-tf2.git)

### [standoff2conll](https://github.com/spyysalo/standoff2conll)
