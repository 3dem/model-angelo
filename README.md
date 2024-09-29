# ModelAngelo

ModelAngelo is an automatic atomic model building program for cryo-EM maps.

## Compute requirements

It is highly recommended to have access to GPUs with at least 8GB of memory. ModelAngelo performs well on NVIDIA GPUs such as 2080's and beyond.

Please note that the weight files required by both ModelAngelo and the language model it uses combined are around 10 GB. So you need to have more disk space than that.

## Installation
<details>
<summary>Personal use</summary>
<br>

(If you manage a computational cluster, please skip to the next section)

**Step 1: Install Conda**

To install ModelAngelo, you need Anaconda. We recommend installing miniconda3, as it is lighter. You can find that here: [miniconda](https://docs.conda.io/en/latest/miniconda.html)

Once you have a version of Anaconda installed, check that it actually runs with:
```
conda info
```

**Step 2: Clone this repo**

Now, you can install ModelAngelo. First, you need to clone this Github repository with 
```
git clone https://github.com/3dem/model-angelo.git
```

**Step 3: Run install script**

After, all you need to do is go into the `model-angelo` directory and run the install script:
```
cd model-angelo
source install_script.sh
```
You will now have a conda environment called `model_angelo` that is able to run the program. 
You need to activate this conda environment with `conda activate model_angelo`. 
Now, you can run `model_angelo build -h` to see if the installation worked!
<br>
</details>

<details>
<summary>Shared computational environment</summary>
<br>

If you manage a computational cluster with many users and would like to install ModelAngelo once to be used everywhere, 
you should complete the above steps 1 and 2 for a public account.

Next, you should designate a folder to save the shared weights of the model such that it can be *readable and executable*
by all users of your cluster. Let's say that path is `/public/model_angelo_weights`.

Now, you run the following:
```
export TORCH_HOME=/public/model_angelo_weights
cd model-angelo
source install_script.sh --download-weights
```
Once the script is finished running, make sure that where it installed the weights is in the directory you set.

Finally, you can make the following bash script available for all users to run:

```
#!/bin/bash
source `which activate` model_angelo
model_angelo "$@"
```
<br>
</details>

<details>
<summary>Installation issues</summary>
<br>

**1. Binary activate not found**
It appears that miniconda's activate binary is not added to `PATH` by default. You can either fix this by appending it yourself, like so:
```
export PATH="$PATH:/path/to/miniconda3/bin"
```
or running `conda init` and restarting your shell.

</details>

## Usage
First, make sure to run `model_angelo build --help` or `model_angelo build_no_seq --help` to familiarize yourself with all of the options available.

<details>
<summary>Building a map with FASTA sequence</summary>
<br>

This is the recommended use case, when you have access to a medium-high resolution cryo-EM map (resolutions exceeding 4 Å) as well as a FASTA files with all of your protein, RNA, and DNA sequences.

Let's say the map's name is `map.mrc` and the (protein) sequence file is `prot.fasta`. To build your model in a directory named `output`, you run:
```
model_angelo build -v map.mrc -pf prot.fasta -o output
```
If you would like to build nucleotides as well, you need to provide the RNA and DNA portions of your sequences in different files like so
```
model_angelo build -v map.mrc -pf prot.fasta -df dna.fasta -rf rna.fasta -o output
```
If you only have RNA or DNA, you can drop the other input.

If the output of the program halts before the completion of `GNN model refinement, round 3 / 3`, there was a bug that you can see in `output/model_angelo.log`. Otherwise, you can find your model in `output/output.cif`. The name of the mmCIF file is based on the output folder name, so if you specify, for example, `-o testing/test/model_building`, the model will be in `testing/test/model_building/model_building.cif`.
</details>

<details>
<summary>Building a map with no FASTA sequence</summary>
<br>

If you have a sample where you do not know all of the protein sequences that occur in the map, you can run `model_angelo build_no_seq` instead.
This version of the program uses a network that was not trained with input sequences, nor does it do post-processing on the built map.

Instead, in addition to a built model, it provides you with HMM profile files that you can use to search a database such as UniRef with HHblits.

You run this command:
```
model_angelo build_no_seq -v map.mrc -o output
```
The model will be in `output/output.cif` as before. 
</details>

<details>
<summary>Using the HMM search procedure</summary>
<br>
If you have a map and would like to identify new proteins, first build it with the no sequence option as above.
You will have the model as an mmCIF.

Now there are also HMM profiles for each chain in HMMER3 format here: `output/hmm_profiles`.

To do a sequence search, you should first download a fasta file that includes your proteins of interest. Often, this will be the proteome of the organism in question. Here are some example proteomes in a `FASTA` format:

<div align="center">
	
| Organism | Link |
| -------- | ---- |
| *E.coli* (K12) | [Download](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/UP000000625/UP000000625_83333.fasta.gz) |
| Human | [Download](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz) |
| *Saccharomyces cerevisiae* | [Download](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000002311/UP000002311_559292.fasta.gz) |

</div>

After downloading this file, you will need to uncompress it with `gunzip`. Then, you can run

```
model_angelo hmm_search --i output --f PATH_TO_DB --o hmm_output
```
You will have your results as a series of HMM output files with the extension .hhr, for example: `hmm_output/A.hhr`. These are named by chain according to the model built by ModelAngelo in `output/output.cif`.

For further convenience, especially for large maps, we also now have two csv files named `output/all_hits.csv` and `output/best_hits.csv`. The first will have all the hits for all the chains sorted by the E-value. The second has all the *unique* hits, with the information of the best hit.

</details>

## FAQs

1. **How do I change which GPU ModelAngelo runs on?** You can specify the device(s) ModelAngelo runs on by using the `--device` flag. So, for example, to use GPU with Id 0, you write `--device 0`. To use the first two GPUs of your computer, you can write `--device 0,1`.
2. **Do I need to repeat the sequence of a dimer twice in the FASTA file?** No, each *unique* sequence only needs to show up once in the FASTA file. Duplicates are always removed.
3. **How does ModelAngelo deal with glycosylation sites, non standard amino acids, etc?** It *doesn't*. These parts of the model should be checked manually.
4. **How does ModelAngelo deal with cis prolines?** It *doesn't*. However, we find that a round of refinement (with REFMAC, for example) fixes this issue.

## Common issues

1. If the result looks very bad, with many disconnected chains, take a look at the alpha helices. If these are made of short and disconnected chains, the map was probably in the wrong hand. If you flip the map and run again, you should see much better results.
2. If the map is processed using deepEMhancer, we have noticed less than satisfactory results. Please try with a map post-processed with a conventional algorithm and try again.
3. Always check your input sequence files to make sure that they correspond to a correct FASTA format. Please make sure that the sequences are all capital letters, as is the convention.
4. If the output model is shifted with respect to your map, make sure that the map provided to ModelAngelo is cubic. Otherwise, it might get shifted when ModelAngelo internally makes the map cubic.

## Citation
- **ModelAngelo 0.3**: Published in the proceedings of the International Conference on Learning Representations (*ICLR*) 2023. You can find the paper on  [Openreview](https://openreview.net/forum?id=65XDF_nwI61). Bibtex:
```
@inproceedings{jamali2023modelangelo,
  title={A Graph Neural Network Approach to Automated Model Building in Cryo-EM Maps},
  author={Kiarash Jamali and Dari Kimanius and Sjors HW Scheres},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=65XDF_nwI61}
}
``` 
- **ModelAngelo 1.0**: Paper available in [Nature](https://www.nature.com/articles/s41586-024-07215-4) . Bibtex:
```
@article{jamali2024automated,
	author = {Kiarash Jamali and Lukas Kall and Rui Zhang and Alan Brown and Dari Kimanius and Sjors Scheres},
	title = {Automated model building and protein identification in cryo-EM maps},
	year = {2024},
	doi = {10.1038/s41586-024-07215-4},
	URL = {https://www.nature.com/articles/s41586-024-07215-4},
	journal = {Nature}
}
```
