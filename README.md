# RLDS Dataset Modification

This repo contains scripts for modifying existing RLDS datasets. 
By running [`modify_rlds_dataset.py`](modify_rlds_dataset.py), you will load an existing RLDS dataset, apply the specified
modifications to each sample, reshard the resulting dataset and store it in a new directory. Apart from a number of simple
modification functions, this repo implements a parallelized `AdhocTFDSBuilder` that can perform the data modifications
in parallel for increased conversion speed.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow` and `tensorflow_datasets`.

To download datasets from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) Google cloud bucket, 
please install `gsutil` using the [installation instructions](https://cloud.google.com/storage/docs/gsutil_install).


## Modifying RLDS Datasets

The command below resizes all RGB and depth images to a max. size of 336 and encodes RGB images as jpeg. 
This can e.g. be useful for reducing the storage size of datasets in the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/).
```
python3 modify_rlds_dataset.py --dataset=<name_of_your_tfds_dataset> --mods=resize_and_jpeg_encode --target_dir=<path_where_mod_data_is_written>
```

This creates a new dataset with smaller, jpeg encoded images in the `target_dir`. 

You can switch out the `resize_and_jpeg_encode` mod for other functions in [mod_functions.py](rlds_dataset_mod/mod_functions.py).


## Command Arguments

The [`modify_rlds_dataset.py`](modify_rlds_dataset.py) script supports the following command line arguments:
```
modify_rlds_dataset.py:
  --data_dir: Directory where source data is stored.
  --dataset: Dataset name.
  --max_episodes_in_memory: Number of episodes converted & stored in memory before writing to disk.
    (default: '100')
    (an integer)
  --mods: List of modification functions, applied in order.
    (a comma separated list)
  --n_workers: Number of parallel workers for data conversion.
    (default: '10')
    (an integer)
  --target_dir: Directory where modified data is stored.
```
You can increase the `n_workers` and `max_episodes_in_memory` parameters based on the resources of your machine. 
The larger the respective value, the faster the dataset conversion. 

A list of all supported dataset modifications ("mods") can be found in [mod_functions.py](rlds_dataset_mod/mod_functions.py).


## Adding New Mods

You can add your own custom modification functions in [mod_functions.py](rlds_dataset_mod/mod_functions.py) by implementing 
the `TfdsModFunction` interface. Your mod function needs to provide one function that modifies the dataset feature spec
and one map function that modifies an input `tf.data.Dataset`. You can use the existing mod functions as examples.
Make sure to register your new mod in the `TFDS_MOD_FUNCTIONS` object.


## Download Open X-Embodiment Dataset
To download the Open X-Embodiment dataset and convert it for training, run `bash prepare_open_x.sh`. You can
specify the download directory at the top of the script.


## FAQ / Troubleshooting

- **No new tempfile could be created**: The script stores large datasets in intermediate temporary files in the 
`\tmp` directory. Depending on the dataset size it can store up to 1000 such temp files. The default number of 
files openable in parallel in Ubuntu is 1024, so this limit can lead to the error above. You can increase the limit by
running `ulimit -n 200000`.




