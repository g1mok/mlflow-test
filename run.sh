#!/bin/bash

usage(){
    cat <<EOF
Usage: run.sh [-h|--help]
Example
    - run.sh -v v1.0 -t 'first tag on v1' -c 'first commit' -b 's3://ex-dataset-jw/key_data/'

Available options:
-h, --help           Print this help and exit
-v, --version        Set tag version
-t, --tagmsg         Enter tag message
-c, --commit         Enter commit message
-s, --source         Enter S3 source bucket path
EOF
    exit
}


chmod +x ./shell_scripts/init.sh
chmod +x ./shell_scripts/add_data.sh

if [[ $# -eq 0 ]]
then
    usage
fi

if [ $# -gt 0 ]
then
    dvc_dir=".dvc"
    if [ ! -d $dvc_dir ]
    then
    echo ".dvc not exist"
    ./shell_scripts/init.sh
    fi
fi

while [ $# -gt 0 ];
do
    case "$1" in
        -h|--help)
            usage
            exit 0;;
        
        -v|--version)
            if [ -n "$2" ]
            then
                tag_version=$2
                shift 2
            else
                echo "[ERROR] add need 4 parameters(-v, -t, -c, -s)"
                exit
            fi;;
            
        -t|--tagmsg)
            if [ -n "$2" ]
            then
                tag_message=$2
                shift 2
            else
                echo "[ERROR] add need 4 parameters(-v, -t, -c, -s)"
                exit
            fi;;
        -c|--commit)
            if [ -n "$2" ]
            then
                commit_message=$2
                shift 2
            else
                echo "[ERROR] add need 4 parameters(-v, -t, -c, -s)"
                exit
            fi;;
        -s|--source)
            if [ -n "$2" ]
            then
                bucket_path=$2
                shift 2
            else
                echo "[ERROR] add need 4 parameters(-v, -t, -c, -s)"
                exit
            fi;;
        -*)
            echo "[ERROR] Invalid Argument"
            exit 1;;
        *)
            echo "[ERROR] Argument not proper"
            exit 1;;
    esac
done

echo "tag_version: ${tag_version}"
echo "tag_message: ${tag_message}"
echo "commit_message: ${commit_message}"
echo "bucket_path: ${bucket_path}"

echo "data add"
./shell_scripts/add_data.sh "${commit_message}" "${tag_version}" "${tag_message}" "${bucket_path}"
python train.py --config train_config.yaml --tag ${tag_version}