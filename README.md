# CV-coop-contrastive-debias
Debiasing method for computer vision neural models. In this repository we first train a biased classifier (for example purposes only) and then we use our method to mitigate its bias. To do this, we train a autoencoder capable of altering the biased features in the image, enhancing the classifier capability.

## Objective:
* Use a autoencoder with contrastive loss to mitigate computer vision model biases. 

## Datasets:
* CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

<!-- ## Related Work:
* Our work is based on other two papers:
  * [Universal Triggers](https://arxiv.org/abs/1908.07125)
    * Which the code is heavily inspired.
  * [Bias Triggers](https://arxiv.org/abs/2005.00268) -->

## Running the Method:

### Installing dependencies
After cloning the repository, run the following commands on the main folder to create the enviroment and install the dependencies:
```
conda create --name advsbias --file conda-requirements.txt
conda activate advsbias
conda install pip
pip install -r pip-requirements.txt
pip uninstall torch
pip3 install torch torchvision torchaudio
```
### Downloading Dataset
* Go to the CelebA website and follow the instructions to download the aligned dataset. The img_align_celeba.zip is the only file that must be downloaded, and should be placed in the './datasets/celeba' folder. Then, from the same folder,run the following command to unzip the file:
```
unzip img_align_celeba.zip
```
Both the unzipped folder an the original .zip file must be kept. Other necessary information to run this method on the CelebA dataset is already included in the './datasets/celeba' folder.

### Training the biased classifier

* To train a biased classifier for the CelebA dataset, you may include any of the 40 CelebA attributes as the target (--target) for the classifier and another as the biased (--bias) correlation. The --bias_prop argument controls how much of the dataset won't be correlated. Some scripts examples are provided in the './scripts/classifier' folder. You may use the following command from the main folder to train the classifier:
```
./scripts/classifier/<script_name>.sh
```
The model will be trained and validated on the same split, according the the arguments provided. The best model will lastly be evaluated in another split without the intended bias, and will be saved on './checkpoints/classifier/<args.name>'. The tensorboard runs will be saves on './runs/<args.name>'.

### Using the debiasing autoencoder method

* Similar to the classifier training, you must also provide the --target, --bias and --bias_prop, in order to use the same dataset as in classifier training. You may also set --alpha and --theta which controls how much of the loss is composed of classification and contrastive loss, respectively.Some scripts examples are provided in the './scripts/autoencoder' folder, which include exploration with different values. You may use the following command from the main folder to train the autoencoder:
```
./scripts/autoencoder/<script_name>.sh
```
* If you never trained a autoencoder for this --target, --bias and -bias_prop, the script will automatically train a autoencoder for reconstruction only, which will be used as a starting point for the debiasing autoencoder and for future runs with these arguments.

* The best model will be saved on './checkpoints/autoencoder/<args.name>'. The tensorboard runs will be saves on './runs_ae/<args.name>'.
 