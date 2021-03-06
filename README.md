# nn-avx2
Neural Network implementation (Back/Forward)with avx2 and fma(4) instructions. This is a full batch app, so everything loaded into memory. Currently best performs on single hidden layer (input + 1 hidden + output) yet still hidden node counts are configurable.

This is my C application I have used for kaggle competitions. Input layer needs to be space separated, see xShuffled.dat and yShuffled.day for example. It uses clang,pthread and -mfma -mavx2 instructions. You might want to update hardcoded cpu architecture on related files (find . -type f -exec grep -l 'znver1' {} \;) 

Building:
```
cd Release
make clean
make all -j32
```
Running with example data (5000 20x20 digit images, each image contains picture of numbers from 0 to 9):
```
cd Release
nice taskset 0x01 ./nn-avx2 -x ../xShuffled.dat -y ../yShuffled.dat -r 5000 -c 400 -n 10 -t 1 -h 256 -i 513 -l 0.99 -j $(nproc) -test 15 -ps 64
```
This example means:
 Run with high thread priority, run main thread on first core, input layer  400, hidden layer 256, output layer 10 , total 5000 image, iterate 513 times, use all threads, test 15% of data and train 85%, show me steps on each 64th iteration. If thread count more than 1 then rest of the threads will spread from max to low, i.e. if 8 core and and 6 thread requested and main thread set to 0, then 7,6,5,4,3 and 0 will be pinned.

If you would like to watch more steps decrease -ps param to a lower number which is power of 2, i.e. -ps 16 

```
USAGE:

--help	This help info

-x	X(input) file path, space separated

-y	Y(output, expected result) file path

-r	Rowcount of X or Y file (should be equal)

-c	Column count of X file (each row should have same count)

-n	Number of labels in Y file (how many expected result, should be a sequence starting from 1)

-t	Total hidden layer count (excluding input and output), currently only 1, 

-h	Hidden layer size (excluding bias unit)

-j	Number of cores(threads) on host pc

-i	Number of iteration for training

-l	Lambda value, between 0-1

-f	Scale inputs for featured list, 0 or 1, optional, default 0)

-p	Do prediction for each input after training complete (0 for disable 1 for enable default 1)

-tp	Theta path. If you have previously saved a prediction result you can continue
	from this result by loading from file path. (-lt value should be 1)

-lt	Load previously saved thetas (prediction result)
	(0 for disable 1 for enable default 0) (-tp needs to be set)

-st	Save thetas (prediction result)(0 for disable 1 for enable default 1), if ps is low this option might cause lot of saving on weight snapshots.

-test	Test percentage, i.e. for 1000 row of data, -test 10 will result: 900 of row for training and 100 for test

-ps	Prediction step, has to be power of 2, for long running tasks you can enable this and -st parameter. I.e. -ps 16 will result every 16 iteration will run prediction against test and if -st 1 then also weights will be saved for this prediction, that later you can load back

-cpus	Total cpu count on system, if system not able to report total numbers due to isolation set this number to actual total, still -j parameter will be considered but this param will make affinity reliable

```


As a hobby project, I am still trying to improve performance. There is around 12% cache-misses because of weights nested loop tiling:

```

 Performance counter stats for 'nice taskset 0x01 ./nn-avx2 -x ../xShuffled.dat -y ../yShuffled.dat -r 5000 -c 400 -n 10 -t 1 -h 256 -i 320 -l 0.99 -j 32 -test 15 -ps 64 -cpus 32' (3 runs):

        187,380.86 msec task-clock:u              #   17.073 CPUs utilized            ( +-  0.10% )
   738,209,855,154      cycles:u                  #    3.940 GHz                      ( +-  0.11% )  (83.35%)
   810,408,433,351      instructions:u            #    1.10  insn per cycle           ( +-  0.01% )  (83.33%)
    76,676,312,410      cache-references:u        #  409.200 M/sec                    ( +-  0.01% )  (83.34%)
     9,475,489,651      cache-misses:u            #   12.358 % of all cache refs      ( +-  0.31% )  (83.32%)
    65,017,640,158      branches:u                #  346.981 M/sec                    ( +-  0.01% )  (83.32%)
       846,205,276      branch-misses:u           #    1.30% of all branches          ( +-  0.01% )  (83.34%)
            22,129      faults:u                  #    0.118 K/sec                    ( +- 10.79% )
            22,129      minor-faults:u            #    0.118 K/sec                    ( +- 10.79% )
                 0      cs:u                      #    0.000 K/sec
                 0      migrations:u              #    0.000 K/sec

           10.9756 +- 0.0277 seconds time elapsed  ( +-  0.25% )


```

But without tiling 1% cache-miss has added little to no benefit even additional second for the same run, so I stick to nested tiling:

```

 Performance counter stats for 'nice taskset 0x01 ./nn-avx2 -x ../xShuffled.dat -y ../yShuffled.dat -r 5000 -c 400 -n 10 -t 1 -h 256 -i 320 -l 0.99 -j 32 -test 15 -ps 64 -cpus 32' (3 runs):

        187,782.20 msec task-clock:u              #   16.150 CPUs utilized            ( +-  0.24% )
   739,271,525,396      cycles:u                  #    3.937 GHz                      ( +-  0.27% )  (83.33%)
   921,219,541,904      instructions:u            #    1.25  insn per cycle           ( +-  0.00% )  (83.32%)
    78,688,675,909      cache-references:u        #  419.042 M/sec                    ( +-  0.33% )  (83.35%)
     1,449,044,455      cache-misses:u            #    1.841 % of all cache refs      ( +-  1.72% )  (83.33%)
    99,109,420,285      branches:u                #  527.789 M/sec                    ( +-  0.00% )  (83.34%)
     1,466,843,024      branch-misses:u           #    1.48% of all branches          ( +-  0.49% )  (83.33%)
            26,250      faults:u                  #    0.140 K/sec                    ( +-  4.47% )
            26,250      minor-faults:u            #    0.140 K/sec                    ( +-  4.47% )
                 0      cs:u                      #    0.000 K/sec
                 0      migrations:u              #    0.000 K/sec

            11.627 +- 0.272 seconds time elapsed  ( +-  2.34% )



```
