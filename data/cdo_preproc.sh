# check if environment variables are set
if [ -z "$REPO_ROOT_DIR" ]; then
  echo "REPO_ROOT_DIR is not set. Please set REPO_ROOT_DIR to the root directory of the repository."
  exit 1
fi
if [ -z "$DATA_ROOT_DIR" ]; then
  echo "DATA_ROOT_DIR is not set. Please set DATA_ROOT_DIR to the root directory of the data."
  exit 1
fi

# -----------------------

echo "Begin"
timestampstr=$(date +"%Y%m%d%H%M%S")
echo "Timestamp: $timestampstr"


repo_root_dir=$REPO_ROOT_DIR
data_root_dir=$DATA_ROOT_DIR

logfilepath="$repo_root_dir/data/logs/cdo_preproc_$timestampstr.log"
echo "Begin" > $logfilepath
echo "Timestamp: $timestampstr" > $logfilepath

cdoo='/usr/bin/cdo'
llsdapy='/usr/bin/python'
xrscript="$repo_root_dir/data/xarray_preproc.py"

# Set data dir
C_DATA_DIR="$data_root_dir/COSMO_PATCH/COSMO_PATCH_2006-2019/data"
ls -lsah $C_DATA_DIR

# Set merged-data dir
C_MERGED_DIR="$C_DATA_DIR/merged_psl-tas-uas-vas"
if [ ! -d "$C_MERGED_DIR" ]; then
  echo "Create directory $C_MERGED_DIR" >> $logfilepath
  mkdir -p "$C_MERGED_DIR"
  for var in "psl" "tas" "uas" "vas"; do
    echo "Create directory $C_MERGED_DIR/$var" >> $logfilepath
    mkdir -p "$C_MERGED_DIR/$var"
  done
else
  echo "Directory $C_MERGED_DIR already exists." >> $logfilepath
fi

# Set data dir for training data (2006-2013)
C_TRAIN_DIR="$data_root_dir/COSMO_PATCH/COSMO_PATCH_2006-2019/train2006-2013/data"
if [ ! -d "$C_TRAIN_DIR" ]; then
  echo "Create directory $C_TRAIN_DIR" >> $logfilepath
  mkdir -p "$C_TRAIN_DIR"
  for var in "psl" "tas" "uas" "vas"; do
    echo "Create directory $C_TRAIN_DIR/$var" >> $logfilepath
    mkdir -p "$C_TRAIN_DIR/$var"
  done
else
  echo "Directory $C_MERGED_DIR already exists." >> $logfilepath
fi

# Set data dir for test data (2014-2018)
C_TEST_DIR="$data_root_dir/COSMO_PATCH/COSMO_PATCH_2006-2019/test2014-2018/data"
if [ ! -d "$C_TEST_DIR" ]; then
  echo "Create directory $C_TEST_DIR" >> $logfilepath
  mkdir -p "$C_TEST_DIR"
  for var in "psl" "tas" "uas" "vas"; do
    echo "Create directory $C_TEST_DIR/$var" >> $logfilepath
    mkdir -p "$C_TEST_DIR/$var"
  done
else
  echo "Directory $C_MERGED_DIR already exists." >> $logfilepath
fi

# Set data dir for stats (computed on training data)
C_STATS_DIR="$data_root_dir/COSMO_PATCH/COSMO_PATCH_2006-2019/train2006-2013/stats"
if [ ! -d "$C_STATS_DIR" ]; then
  echo "Create directory $C_STATS_DIR" >> $logfilepath
  mkdir -p "$C_STATS_DIR"
  for var in "psl" "tas" "uas" "vas"; do
    echo "Create directory $C_STATS_DIR/$var" >> $logfilepath
    mkdir -p "$C_STATS_DIR/$var"
  done
else
  echo "Directory $C_STATS_DIR already exists." >> $logfilepath
fi


# BEGIN

echo "" >> $logfilepath
echo "I. Merging all files" >> $logfilepath
echo "--------------------" >> $logfilepath

for var in "psl" "tas" "uas" "vas"; do
  echo "$var" >> $logfilepath

  if [ -f "$C_MERGED_DIR/$var/${var}_raw_merged.nc" ]; then
    echo "File ${var}_raw_merged.nc already exists." >> $logfilepath
  else
    echo "Merge all files." >> $logfilepath
    echo " > cdo mergetime $C_DATA_DIR/$var/${var}_EUR-6km_ECMWF-ERAINT_REA6_r1i1p1f1_COSMO_v1_*.nc $C_MERGED_DIR/$var/${var}_raw_merged.nc" >> $logfilepath
    $cdoo mergetime \
      $C_DATA_DIR/$var/${var}_EUR-6km_ECMWF-ERAINT_REA6_r1i1p1f1_COSMO_v1_*.nc \
      $C_MERGED_DIR/$var/${var}_raw_merged.nc
  fi
done


echo "" >> $logfilepath
echo "II.i. Selecting training data (2006-2013)" >> $logfilepath
echo "-----------------------------------------" >> $logfilepath

for var in "psl" "tas" "uas" "vas"; do
  echo "$var" >> $logfilepath

  if [ -f "$C_TRAIN_DIR/$var/${var}_raw_merged.nc" ]; then
    echo "File ${var}_raw_merged.nc already exists." >> $logfilepath
  else
    echo "Select years 2006 - 2013." >> $logfilepath
    echo " > cdo seldate,2006-01-01T01:00:00,2013-12-31T23:00:00 $C_MERGED_DIR/$var/${var}_raw_merged.nc $C_TRAIN_DIR/$var/${var}_raw_merged.nc" >> $logfilepath
    $cdoo seldate,2006-01-01T01:00:00,2013-12-31T23:00:00 \
      $C_MERGED_DIR/$var/${var}_raw_merged.nc \
      $C_TRAIN_DIR/$var/${var}_raw_merged.nc
  fi
done


echo "" >> $logfilepath
echo "II.ii. Selecting test data (2014-2018)" >> $logfilepath
echo "--------------------------------------" >> $logfilepath

for var in "psl" "tas" "uas" "vas"; do
  echo "$var" >> $logfilepath

  if [ -f "$C_TEST_DIR/$var/${var}_raw_merged.nc" ]; then
    echo "File ${var}_raw_merged.nc already exists." >> $logfilepath
  else
    echo "Select years 2014 - 2018." >> $logfilepath
    echo " > cdo seldate,2014-01-01T00:00:00,2018-12-31T23:00:00 $C_MERGED_DIR/$var/${var}_raw_merged.nc $C_TEST_DIR/$var/${var}_raw_merged.nc" >> $logfilepath
    $cdoo seldate,2014-01-01T00:00:00,2018-12-31T23:00:00 \
      $C_MERGED_DIR/$var/${var}_raw_merged.nc \
      $C_TEST_DIR/$var/${var}_raw_merged.nc
  fi
done


echo "" >> $logfilepath
echo "III. Compute quantiles" >> $logfilepath
echo "--------------------" >> $logfilepath

for var in "psl" "tas" "uas" "vas"; do
  echo "$var" >> $logfilepath

  if [ -f "$C_STATS_DIR/$var/${var}_quantiles.nc" ]; then
    echo "File ${var}_quantiles.nc already exists." >> $logfilepath
  else
    echo "Compute quantiles." >> $logfilepath
    echo "> python $xrscript quantiles $C_TRAIN_DIR/$var/${var}_raw_merged.nc $C_STATS_DIR/$var/${var}_quantiles.nc" >> $logfilepath
    $llsdapy $xrscript quantiles\
      $C_TRAIN_DIR/$var/${var}_raw_merged.nc \
      $C_STATS_DIR/$var/${var}_quantiles.nc
  fi
done


echo "" >> $logfilepath
echo "IV.i. Merge datasets (train)" >> $logfilepath
echo "----------------------------" >> $logfilepath

if [ ! -d "$C_TRAIN_DIR/allvars" ]; then
  echo "Create directory $C_TRAIN_DIR/allvars" >> $logfilepath
  mkdir -p "$C_TRAIN_DIR/allvars"
else
  echo "Directory $C_TRAIN_DIR/allvars already exists." >> $logfilepath
fi

if [ -f "$C_TRAIN_DIR/allvars/merged-allvars.nc" ]; then
  echo "File merged-allvars.nc already exists." >> $logfilepath
else
  echo "Merge all variables." >> $logfilepath

  echo "> cdo merge $C_TRAIN_DIR/psl/psl_raw_merged.nc $C_TRAIN_DIR/tas/tas_raw_merged.nc $C_TRAIN_DIR/uas/uas_raw_merged.nc $C_TRAIN_DIR/vas/vas_raw_merged.nc $C_TRAIN_DIR/allvars/merged-allvars.nc" >> $logfilepath
  $cdoo merge \
    $C_TRAIN_DIR/psl/psl_raw_merged.nc \
    $C_TRAIN_DIR/tas/tas_raw_merged.nc \
    $C_TRAIN_DIR/uas/uas_raw_merged.nc \
    $C_TRAIN_DIR/vas/vas_raw_merged.nc \
    $C_TRAIN_DIR/allvars/merged-allvars.nc
fi



echo "" >> $logfilepath
echo "IV.ii. Merge datasets (test)" >> $logfilepath
echo "----------------------------" >> $logfilepath

if [ ! -d "$C_TEST_DIR/allvars" ]; then
  echo "Create directory $C_TEST_DIR/allvars" >> $logfilepath
  mkdir -p "$C_TEST_DIR/allvars"
else
  echo "Directory $C_TEST_DIR/allvars already exists." >> $logfilepath
fi

if [ -f "$C_TEST_DIR/allvars/merged-allvars.nc" ]; then
  echo "File merged-allvars.nc already exists." >> $logfilepath
else
  echo "Merge all variables." >> $logfilepath

  echo "> cdo merge $C_TEST_DIR/psl/psl_raw_merged.nc $C_TEST_DIR/tas/tas_raw_merged.nc $C_TEST_DIR/uas/uas_raw_merged.nc $C_TEST_DIR/vas/vas_raw_merged.nc $C_TEST_DIR/allvars/merged-allvars.nc" >> $logfilepath
  $cdoo merge \
    $C_TEST_DIR/psl/psl_raw_merged.nc \
    $C_TEST_DIR/tas/tas_raw_merged.nc \
    $C_TEST_DIR/uas/uas_raw_merged.nc \
    $C_TEST_DIR/vas/vas_raw_merged.nc \
    $C_TEST_DIR/allvars/merged-allvars.nc
fi


echo "" >> $logfilepath
echo "V. Merge stats" >> $logfilepath
echo "-------------------" >> $logfilepath

if [ ! -d "$C_STATS_DIR/allvars" ]; then
  echo "Create directory $C_STATS_DIR/allvars" >> $logfilepath
  mkdir -p "$C_STATS_DIR/allvars"
else
  echo "Directory $C_STATS_DIR/allvars already exists." >> $logfilepath
fi


if [ -f "$C_STATS_DIR/allvars/merged-allvars_quantiles.nc" ]; then
  echo "File merged-allvars_quantiles.nc already exists." >> $logfilepath
else
  echo "Merge quantiles of all variables." >> $logfilepath
  echo "> cdo merge $C_STATS_DIR/psl/psl_quantiles.nc $C_STATS_DIR/tas/tas_quantiles.nc $C_STATS_DIR/uas/uas_quantiles.nc $C_STATS_DIR/vas/vas_quantiles.nc $C_STATS_DIR/allvars/merged-allvars_quantiles.nc" >> $logfilepath
  $cdoo merge \
    $C_STATS_DIR/psl/psl_quantiles.nc \
    $C_STATS_DIR/tas/tas_quantiles.nc \
    $C_STATS_DIR/uas/uas_quantiles.nc \
    $C_STATS_DIR/vas/vas_quantiles.nc \
    $C_STATS_DIR/allvars/merged-allvars_quantiles.nc
fi




echo "Done" >> $logfilepath