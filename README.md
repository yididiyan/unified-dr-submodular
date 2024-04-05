## unified-dr-submodular
This repository contains implementation for "**Pedramfar M, Nadew YY, Quinn CJ, Aggarwal V. Unified Projection-Free Algorithms for Adversarial DR-Submodular Optimization. To appear in The Twelfth International Conference on Learning Representations (ICLR 2024)**". 
A preprint is available at [arXiv](https://arxiv.org/abs/2403.10063).

## Setting up virtual environment using conda 
Install Python 3.8.10 using conda 

```bash
# Setup virtual environment 
conda create --name dr_submod python=3.8.10
# Activate environment 
conda activate dr_submod
```



## Installation 
### Installing dependencies 
```bash
pip install -r requirements.txt
```

### Installing dr_submodular as a module   
```
pip install -e .
```



### Running experiments

To run quadratic experiments included in the paper, navigate to `./experiments/quadratic_programming` directory. 

```bash
cd ./experiments/quadratic_programming 

```

#### GMFW and SMFW algorithms 
The following will run the GMFW and SBFW algorithms on synthetic experiments with multiple seeds. In the command,
* `Ts` are the list of horizons.
* `seeds` are list of seed values to randomize experiments.
* `h-scale` refers to the scale of the quadratic coefficients, and  
* `grad-noise` is the scale of the normalized gradient noise.   

```bash
python experiments.py --Ts <time-horizons> --seeds <seeds> --h-scale 10. --grad-noise 0.1 
```

### Plotting results 

```bash 
python plot.py --Ts <time-horizons>  --seeds <seeds>
```

This produces instantaneous and cumulative regret plot for the above experiments under `experiments/quadratic_programming/plots/`

To reproduce the results in the paper (Figure 2), replace 
`<time-horizons>` with `20 40 80 160 320 500` and `<seeds>` to `1 2 3 4 5 6 7 8 9 10`.  


### Acknowledgement

We thank authors of "Zhang Q, Deng Z, Chen Z, Zhou K, Hu H, Yang Y. Online Learning for Non-monotone DR-Submodular Maximization: From Full Information to Bandit Feedback. International Conference on Artificial Intelligence and Statistics 2023 Apr 11 (pp. 3515-3537). PMLR." for discussion and providing their implementation on parts of which this one is built.  
