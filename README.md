# Adaptive Norms for Deep Learning with Regularized Newton Methods
In this paper, we propose second-order variants of adaptive gradient methods like Adam and RMSProp. We theoretically prove convergence and show their empirical superiority over spherical trust region methods.


![Empirical results](/intro.png)


## Installation

1. Make sure virtualenv is installed \
`pip3 install --user virtualenv`


2. Create a new virtual environment called "venv_etr"\
`python3 -m virtualenv venv_etr`

3. Activate the virtual environment\
`source venv_etr/bin/activate`

4. install etr and its requirements by running the following command inside the root folder\
`pip install -r requirements.txt`

5. Install Jupyter Kernel\
`ipython kernel install --user --name=venv_etr`

## Run demo
6. Start Jupyter Notebook Demo\
`jupyter notebook` -> then go on demo.ipynb

7. Change to correct kernel\
Under Kernel>Change kernel select venv_etr. Now follow the instructions in the demo notebook.
