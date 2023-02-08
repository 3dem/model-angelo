# ModelAngelo

ModelAngelo is an automatic atomic model building program for cryo-EM maps.

## Compute requirements
It is highly recommended to have access to GPUs with at least 8GB of memory. ModelAngelo performs well on NVIDIA GPUs such as 2080's and beyond.

Please note that the weight files required by both ModelAngelo and the language model it uses combined are around 10 GB. So you need to have more disk space than that.

## Installation
### Personal use

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

### Installing for a shared computational environment
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

## Installation issues

**1. Binary activate not found**
It appears that miniconda's activate binary is not added to `PATH` by default. You can either fix this by appending it yourself, like so:
```
export PATH="$PATH:/path/to/miniconda3/bin"
```
or running `conda init` and restarting your shell.

## Usage
### Building a map with FASTA sequence
This is the recommended use case, when you have access to a medium-high resolution cryo-EM map (resolutions exceeding 4 Å) as well as a FASTA file with all of your protein sequences.

To familiarize yourself with the options available in `model_angelo build`, run `model_angelo build -h`.

Let's say the map's name is `map.mrc` and the sequence file is `sequence.fasta`. To build your model in a directory named `output`, you run:
```
model_angelo build -v map.mrc -f sequence.fasta -o output
```
If the output of the program halts before the completion of `GNN model refinement, round 3 / 3`, there was a bug that you can see in `output/model_angelo.log`. Otherwise, you can find your model in `output/output.cif`. The name of the mmCIF file is based on the output folder name, so if you specify, for example, `-o testing/test/model_building`, the model will be in `testing/test/model_building/model_building.cif`.

### Building a map with no FASTA sequence
If you have a sample where you do not know all of the protein sequences that occur in the map, you can run `model_angelo build_no_seq` instead.
This version of the program uses a network that was not trained with input sequences, nor does it do post-processing on the built map.

Instead, in addition to a built model, it provides you with HMM profile files that you can use to search a database such as UniRef with HHblits.

You run this command:
```
model_angelo build_no_seq -v map.mrc -o output
```
The model will be in `output/output.cif` as before. Now there are also HMM profiles for each chain in HHsearch's format here: `output/hmm_profiles`.
To do a sequence search for chain A (for example), you should first install [HHblits](https://github.com/soedinglab/hh-suite) and download one of the [databases](https://github.com/soedinglab/hh-suite#available-databases). Then, you can run
```
hhblits -i output/hmm_profiles/A.hhm -d PATH_TO_DB -o A.hhr -oa3m A.a3m -M first
```
You will have your result as a multiple sequence alignment here: `A.a3m`. 

## Common issues
1. ModelAngelo currently does not build nucleotides. It also may make mistakes if nucleotide sequences are in the sequence fasta file.

2. If the result looks very bad, with many disconnected chains, take a look at the alpha helices. If these are made of short and disconnected chains, the map was probably in the wrong handedness. If you flip the map and run again, you should see much better results.

## Citation

ModelAngelo has been published in the proceedings of the International Conference on Learning Representations (*ICLR*) 2023. You can find the paper on  [Openreview](https://openreview.net/forum?id=65XDF_nwI61)

Here is the BibTex
```
@inproceedings{
jamali2023modelangelo,
title={A Graph Neural Network Approach to Automated Model Building in Cryo-EM Maps},
author={Kiarash Jamali and Dari Kimanius and Sjors HW Scheres},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=65XDF_nwI61}
}
```
