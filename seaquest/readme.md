Go to the seaquest folder and run the following commands to install the required packages

conda create --name seaquest python=3.8

conda activate seaquest

Install torch in conda. Command can be found at https://pytorch.org/get-started/locally/

pip install git+https://github.com/takuseno/d4rl-atari
pip install "gym[atari, accept-rom-license]"
pip install pyclustering
pip install seaborn
pip install d3rlpy==1.1.1