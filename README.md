# codes-red

To build binary modules use
```bash
make
```

Now experiments can be performed by
```
parallel -j 8 :::: experiments/pot_BKZ_1280.sh
parallel -j 8 :::: experiments/deep_BKZ_1280.sh
parallel -j 8 :::: experiments/basic_BKZ_1280.sh
```

# Requirements
* NumPy 2.0+

* Gap and Guava: for computation of minimum weight codewords. For sage installed with conda/mamba and Ubuntu:
```
sudo apt install gap-guava
conda install gap
```
