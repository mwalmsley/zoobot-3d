#!/bin/sh
for f in data/gz3d/fits_gz/*.gz
do
    old_dir_and_fits=${f%.*}
    new_dir_and_fits=data/gz3d/fits/${old_dir_and_fits##*/}
    # echo $new_dir_and_fits
    do gunzip -c "$f" > new_dir_and_fits
done

# also works, but way less obvious than python version