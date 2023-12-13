# Switchback Statistics
Repository for codebase to analyze switchback statistics to arrive at a consistent definition.

## Environment installation
After having struggled with installing `pysedas` I thought its best if I create a separate `pysedas_env.yml` file to use a different virtual environment when using `pysedas`. The following `mamba` syntax enables to install the packages with existing `conda/mamba` channels (see the packages under `dependencies` in the environment file).
```
mamba env create --file pyspedas_env.yml
```
For me, the above syntax ignores the packages under `pip` and so we have to install them manually after activating the environment as follows
```
conda activate pysedas
pip install pytplot-mpl-temp==2.2.1
pip install geopack==1.0.10
pip install pyspedas
```
