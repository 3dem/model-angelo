#!/bin/bash
ENVNAME=model_angelo
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Make sure you have conda installed"
      echo "Make sure you have set the TORCH_HOME environment variable to a suitable public location (if installing on a cluster)"
      echo "-h, --help                   simple help and instructions"
      echo "-w, --download-weights       use if you want to also download the weights"
      echo "-n, --name                   name of model-angelo conda environment, default: model_angelo"
      exit 0
      ;;
    -w|--download-weights)
      echo "Downloading weights as well because flag -w or --download-weights was specified"
      DOWNLOAD_WEIGHTS=1
      shift
      ;;
    -n|--name)
      ENVNAME="$2"
      echo "Environment Name is: $ENVNAME"
      shift 2
      ;;
  esac
done

if [ -z "${TORCH_HOME}" ] && [ -n "${DOWNLOAD_WEIGHTS}" ]; then
  echo "ERROR: TORCH_HOME is not set, but --download-weights or -w flag is set";
  echo "Please specify TORCH_HOME to a publicly available directory";
  exit 1;
fi

is_conda_model_angelo_installed=$(conda info --envs | grep $ENVNAME -c)
if [[ "${is_conda_model_angelo_installed}" == "0" ]];then
  conda create -n $ENVNAME python=3.10 -y;
fi

torch_home_path="${TORCH_HOME}"

if [[ `command -v activate` ]]
then
  source `which activate` $ENVNAME
else
  conda activate $ENVNAME
fi
  
# Check to make sure model_angelo is activated
if [[ "${CONDA_DEFAULT_ENV}" != $ENVNAME ]]
then
  echo "Could not run conda activate model_angelo, please check the errors";
  exit 1;
fi

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

if [ "${torch_home_path}" ]
then
  conda env config vars set TORCH_HOME="${torch_home_path}"
fi

python_exc="${CONDA_PREFIX}/bin/python"

$python_exc -mpip install .

if [[ "${DOWNLOAD_WEIGHTS}" ]]; then
  echo "Writing weights to ${TORCH_HOME}"
  model_angelo setup_weights --bundle-name nucleotides
  model_angelo setup_weights --bundle-name nucleotides_no_seq
else
  echo "Did not download weights because the flag -w or --download-weights was not specified"
fi
