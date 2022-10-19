# A Deep Learning Based Method to Delineate the Wet/Dry Shoreline and Compute its Elevation Using High-Resolution UAS Imagery
## Content
This repository contains the code used in the described paper

### Publication
Publication under review

Citation:<br>
<i>Regular text citation</i>

<details>
  <summary>LaTeX Citation</summary>
  
        @article{vicensmiquel,
                title={A Deep Learning Based Method to Delineate the Wet/Dry Shoreline and Compute its Elevation Using High-Resolution UAS Imagery},
                author={Vicens-Miquel, Marina and Medrano, Antonio and Tissot, Philippe and Kamangir, Hamid and Starek, Michael},
                journal={Remote Sensing MDPI(Submitted, pending review)}
        }
</details>

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
6) Install packages using pip <i>(using the double equals command to install these versions)</i></br>
        &nbsp; &nbsp; keras = 2.6.0    </br>
        &nbsp; &nbsp; matplotlib = 3.5.1    </br>
        &nbsp; &nbsp; pandas = 1.4.1   </br>
        &nbsp; &nbsp; numpy = 1.19.5   </br>
        &nbsp; &nbsp; scikit-learn = 1.0.2 </br>


### Quickstart
To train the model, please make sure to first activate the environment. After, please type the following command to start training `python src/train.py`



### Data Format
The dataset is organized under the below directories. `testingOrtho1` is the name of the dataset used. In this case, `testingOrtho1` means that we are using the orthomosaic 1 as an independent testing dataset, while using all the other orthomosaic for training and validation. Depending on which orthomosaic we want as the independent testing dataset, a different dataset will be loaded.
		
	data
	└── testingOrtho1
	    ├── training
	    │   ├── original (this directory contains all the original images) 
	    │   └── labeled  (this directory contains all the labeled images)
	    ├── testing
	    │   ├── original (this directory contains all the original images) 
	    │   └── labeled  (this directory contains all the labeled images)
	    └── validation
	        ├── original (this directory contains all the original images) 
	        └── labeled  (this directory contains all the labeled images)

