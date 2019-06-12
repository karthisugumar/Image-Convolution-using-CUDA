# ./convSeq | grep -Eo '[+-]?[0-9]+([.][0-9]+)?'

nvcc -o convConstMem convConstMem.cu --disable-warnings -std=c++11
avg_total=0.0
avg_gpu=0.0
for j in {1..20}
    do
	printf "For iteration $j\n"
	./convConstMem $1 $2 > temp.txt
	time_gpu="$(awk '/\GPU/' temp.txt | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')"
        time_total="$(awk '/\data/' temp.txt | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')"
	printf "Total GPU time: $time_gpu ms\n"
	printf "Total time: $time_total ms\n\n"
	avg_total=$(echo "$avg_total + $time_total" | bc)
	avg_gpu=$(echo "$avg_gpu + $time_gpu" | bc)
	
    done

avg_total=$( echo "scale=6; $avg_total / 20.0" | bc )
avg_gpu=$( echo "scale=6; $avg_gpu / 20.0" | bc )

printf "\n"
printf "\nAverage GPU Execution Time(20 runs)= $avg_gpu"
printf "\nAverage TOTAL Execution Time(20 runs)= $avg_total"
printf "\n"

