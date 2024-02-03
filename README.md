# Reproducibility study of "Explaining RL decisions with trajectories"
This study reproduces the results and adds additional experiments to the paper by Deshmukh et al. (2023).

Shripad Vilasrao Deshmukh, Arpan Dasgupta, Balaji Krishnamurthy, Nan Jiang, Chirag Agarwal, Georgios
Theocharous, and Jayakumar Subramanian. Explaining rl decisions with trajectories, 2023.

## To run the code: 
### Seaquest 
Run the following commands to install the required packages.

~~~sh
cd seaquest
conda create --name seaquest python=3.8
conda activate seaquest
~~~

Install torch in conda. Command can be found at https://pytorch.org/get-started/locally/
Additionally, import d4rl-atari, gym, pyclustering, seaborn and d3rlpy==1.1.1

~~~shell
pip install git+https://github.com/takuseno/d4rl-atari 
pip install "gym[atari, accept-rom-license]" 
pip install pyclustering 
pip install seaborn 
pip install d3rlpy==1.1.1
~~~
Alternatively, build from the environment.yml:
~~~shell
cd seaquest
conda env create -f environment.yml
conda activate seaquest
~~~~
Then run the main file to get the results.
~~~shell
python3 trajectory_attribution.py
~~~

### HalfCheetah 
Build from the environment_HC.yml and install and install the correct GCC:
~~~shell
cd halfcheetah
conda env create -f environment_HC.yml
conda activate halfcheetah
conda install -c conda-forge libstdcxx-ng
~~~~
Then run the main file to get the results.
~~~shell
python3 experiment.py --dataset halfcheetah-medium-v2     --gpt_loadpath gpt/pretrained
~~~

