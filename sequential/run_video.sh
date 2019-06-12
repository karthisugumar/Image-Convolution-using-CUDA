search_dir=/home/ksugumar/project/videos/gtav_FINAL/
out_dir=/home/ksugumar/project/videos/gtav_OUT/

nvcc -o convSeq convSeq.cu -std=c++11 --disable-warnings

count=0
STARTTIME=$(date +%s%N)
totalTime=0.0

for entry in "$search_dir"/*
do
	count=$(($count+1))
	echo "Doing image $count --"
	filename=$(basename $entry)
	totalTime=$( echo "$totalTime + $(./convSeq $search_dir$filename $out_dir$filename | grep -Eo '[+-]?[0-9]+([.][0-9]+)?') " | bc )
	printf "Total Time Taken is: $totalTime\n"
	
done

ENDTIME=$(date +%s%N)

printf "\n\nIt took $totalTime ms to complete this task of converting 950 images...."
printf "\nIt took $((($ENDTIME - $STARTTIME)/1000000)) ms to complete the script...."
