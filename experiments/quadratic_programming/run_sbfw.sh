
for T in 20 40 80 160 320 500
    do
     python experiments.py --alg sbfw --T $T --h-scale 10.0 --grad-noise $1 --seed $2;
done
