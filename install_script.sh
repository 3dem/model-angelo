while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Make sure you have conda installed"
      echo "Make sure you have set the TORCH_HOME environment variable to a suitable public location (if installing on a cluster)"
      echo "-h, --help                   simple help and instructions"
      echo "-w, --download-weights       use if you want to also download the weights"
      exit 0
      ;;
    -w|--download-weights)
      echo "Downloading weights as well because flag -w or --download-weights was specified"
      DOWNLOAD_WEIGHTS=1
      shift
      ;;
  esac
done

if [ -z "${TORCH_HOME}" ] && [ -n "${DOWNLOAD_WEIGHTS}" ]; then
  echo "ERROR: TORCH_HOME is not set, but --download-weights or -w flag is set";
  echo "Please specify TORCH_HOME to a publicly available directory";
  exit 1;
fi

is_conda_model_angelo_installed=$(conda info --envs | grep model_angelo -c)
if [[ "${is_conda_model_angelo_installed}" == "0" ]];then
  conda create -n model_angelo python=3.10 -y;
fi

torch_home_path="${TORCH_HOME}"

if [[ `command -v activate` ]]
then
  source `which activate` model_angelo
else
  conda activate model_angelo
fi
  
# Check to make sure model_angelo is activated
if [[ "${CONDA_DEFAULT_ENV}" != "model_angelo" ]]
then
  echo "Could not run conda activate model_angelo, please check the errors";
  exit 1;
fi

$python_exc -mpip install torch torchvision torchaudio

if [ "${torch_home_path}" ]
then
  conda env config vars set TORCH_HOME="${torch_home_path}"
fi

python_exc="${CONDA_PREFIX}/bin/python"

$python_exc -mpip install -e .

if [[ "${DOWNLOAD_WEIGHTS}" ]]; then
  echo "Writing weights to ${TORCH_HOME}"
  model_angelo setup_weights --bundle-name nucleotides
  model_angelo setup_weights --bundle-name nucleotides_no_seq
else
  echo "Did not download weights because the flag -w or --download-weights was not specified"
fi
