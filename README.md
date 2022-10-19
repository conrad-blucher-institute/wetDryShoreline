# A Deep Learning Based Method to Delineate the Wet/Dry Shoreline and Compute its Elevation Using High-Resolution UAS Imagery
## Contents
This repoitory contains the code used in the described paper

### Publication
Publication under review

Citation:

        @article{vicensmiquel,
                title={A Deep Learning Based Method to Delineate the Wet/Dry Shoreline and Compute its Elevation Using High-Resolution UAS Imagery},
                author={Vicens-Miquel, Marina and Medrano, Antonio and Tissot, Philippe and Kamangir, Hamid and Starek, Michael},
                journal={Remote Sensing MDPI(Submitted, pending review)}
        }


### Installation
1) Install miniconda
        wget https://docs.conda.io/en/latest/miniconda.html
        ./Miniconda3-latest-Linux-x86_64.sh
2) Install mamba
        conda install mamba -n base -c conda-forge
3) Create an environment
        conda create --name tf_gpu 
4) Activate the environment
        conda activate tf_gpu
5) Install TensorFlow
        mamba install tensorflow-gpu -c conda-forge
6) Install packages using pip
        keras==2.6.0    
        matplotlib==3.5.1    
        pandas==1.4.1   
        numpy==1.19.5   
        scikit-learn==1.0.2 

