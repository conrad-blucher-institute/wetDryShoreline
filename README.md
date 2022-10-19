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
1) Install miniconda </br>
        &nbsp; &nbsp; wget https://docs.conda.io/en/latest/miniconda.html </br>
        &nbsp; &nbsp; ./Miniconda3-latest-Linux-x86_64.sh </br>
2) Install mamba </br>
        &nbsp; &nbsp; conda install mamba -n base -c conda-forge </br>
3) Create an environment </br>
        &nbsp; &nbsp; conda create --name tf_gpu  </br>
4) Activate the environment </br>
        &nbsp; &nbsp; conda activate tf_gpu </br>
5) Install TensorFlow </br>
        &nbsp; &nbsp; mamba install tensorflow-gpu -c conda-forge </br>
6) Install packages using pip </br>
        &nbsp; &nbsp; keras==2.6.0    </br>
        &nbsp; &nbsp; matplotlib==3.5.1    </br>
        &nbsp; &nbsp; pandas==1.4.1   </br>
        &nbsp; &nbsp; numpy==1.19.5   </br>
        &nbsp; &nbsp; scikit-learn==1.0.2 </br>

