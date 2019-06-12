search_dir=/home/ksugumar/project/videos/gtav_FINAL/
out_dir=/home/ksugumar/project/videos/gtav_OUT_GPU/

nvcc -o convConstMem convConstMem.cu -std=c++11 --disable-warnings

count=0
STARTTIME=$(date +%s%N)

for entry in "$search_dir"/*
do
	count=$(($count+1))
	echo "Doing image $count --"
	filename=$(basename $entry)
	echo "$((./convConstMem $search_dir$filename $out_dir$filename | awk '/\data/' temp.txt | grep -Eo '[+-]?[0-9]+([.][0-9]+)?'))"
done


ENDTIME=$(date +%s%N)

printf "\n\nIt took $((($ENDTIME - $STARTTIME)/1000000)) ms to complete this task of converting 950 images...."
