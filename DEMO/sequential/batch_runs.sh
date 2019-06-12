## This script runs the convolution 20 times and outputs the average time taken
## Takes in the input image path and the output image path as arguments

nvcc -o convSeq convSeq.cu --disable-warnings -std=c++11
avg=0.0
for j in {1..20}
    do
	printf "For iteration $j\n"
        time="$(./convSeq $1 $2 | awk '/\Total Elapsed/'| grep -Eo '[+-]?[0-9]+([.][0-9]+)?')"
	printf "$time ms\n\n"
	avg=$(echo "$avg + $time" | bc)
    done

avg=$( echo "scale=6; $avg / 20.0" | bc )
printf "\n"
printf "\nAverage Execution Time(20 runs)= $avg\n"

