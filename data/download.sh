#!/bin/bash -eu

version="v1.0"

function download {
    url=https://storage.googleapis.com/ailab-public/webcolor/dataset/${1}
    echo -n downloading ${1##*/} from ${url} ...
    wget ${url} -q || wget ${url}
    echo " done"
}

case ${1:-cache} in
    main )
        download webcolor_${version}.hdf5
        download webcolor_split_${version}.json ;;

    text )
        download webcolor_text_${version}.hdf5 ;;

    image )
        download webcolor_image_${version}.tar.gz ;;

    cache )
        download cache/train_${version}.bin
        download cache/val_${version}.bin
        download cache/test1_${version}.bin
        download cache/test2_${version}.bin
        download webcolor_split_${version}.json ;;

    * )
        echo "Usage: ./download.sh {main,cache,text,image}" ;;
esac
