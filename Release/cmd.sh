xperf stat -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,faults,minor-faults,cs,migrations -r 3 nice taskset 0x01 ./nn-avx2 -x ../xShuffled.dat -y ../yShuffled.dat -r 5000 -c 400 -n 10 -t 1 -h 2048 -i 513 -l 0.99 -j 16 -test 15 -ps 64 -cpus 16 
