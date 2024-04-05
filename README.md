## unified-dr-submodular
This repository contains implementation for Pedramfar M, Nadew YY, Quinn CJ, Aggarwal V. A Unified Approach for Online Continuous DR-Submodular Maximization. InThe Twelfth International Conference on Learning Representations 2023 Oct 13.


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

To run quadratic experiments included in the paper, we run 
Experiments can be found under `./experiments/quadratic_programming` directory. 

```bash
cd ./experiments/quadratic_programming 

```

#### GMFW and SMFW algorithms 
The following will run the GMFW algorithms on 10 synthetic experiments with seeds 1 to 10. In the following commands, 0.1 is the noise scale. 
```bash
for i in {1..10}
do 
   bash run_gmfw.sh 0.1 $i;
   bash run_gmfw.sh 0.1 $i;
done

```

### Acknowledgement

We thank authors of "Zhang Q, Deng Z, Chen Z, Zhou K, Hu H, Yang Y. Online Learning for Non-monotone DR-Submodular Maximization: From Full Information to Bandit Feedback. International Conference on Artificial Intelligence and Statistics 2023 Apr 11 (pp. 3515-3537). PMLR." for discussion and providing their implementation on parts of which this one is built.  
