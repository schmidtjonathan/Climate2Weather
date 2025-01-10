# Climate2Weather: Data Processing

The COSMO REA6 data has to be downloaded and saved in the `DATA_ROOT_DIR` directory.

## Pre-Processing

### COSMO

#### 1. Extracting the spatial region
<p align="center">
<img src="./grid_points.png" width="300">
</p>

The spatial region can be extracted using the `processing.py` script.
For each variable `var`, proceed as follows:
```bash
python processing.py extract_patch --var-folder /path/to/COSMO/var --out-folder /path/to/out/var [--years 2006-2019]
```


#### 2. Processing the patch data

The patch data can be processed using the `cdo_preproc.sh` bash script.
This requires the [`cdo` tool](https://code.mpimet.mpg.de/projects/cdo/wiki) to be installed.
```bash
bash cdo_preproc.sh
```

#### 3. [Optional] Normalize & store the training set as a single HDF5 file

> ⚠️ This step is optional and can/must be changed depending on your preferred data loader. If you want to use the provided data loader, this step needs to be executed.

There are different normalization modes available.
The following command normalizes the data such that 95% (`--quant95`) of the data is within the range [0, 1] and stores the data in a single HDF5 file:
```bash
python processing.py to_normed_h5 --merged-file-path /path/to/merged-allvars.nc --quantiles-filepath /path/to/merged-allvars_quantiles.nc --h5-out-file /path/to/train_norm-quant95.h5 --norm-mode quant95
```

