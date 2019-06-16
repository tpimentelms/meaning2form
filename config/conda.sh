conda create --name syst python=3.6.1
conda activate syst

conda install -y numpy
conda install -y pandas
conda install -y scikit-learn
conda install -y -c conda-forge gensim

conda install -y matplotlib
pip install seaborn

conda install -y tqdm

# conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
