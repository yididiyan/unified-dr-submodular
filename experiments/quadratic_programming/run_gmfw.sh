
for T in 20 40 80 160 320 500;
    do
        for i in 0.0 0.5;
        do
            python experiments.py --alg gmfw --gmfw-beta $i --T $T  --h-scale 10 --grad-noise $1 --seed $2;
        done
done
