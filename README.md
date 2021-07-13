# Dataset Parameters

### Description of CIFAR datasets
These datasets consist of 60,000 images each with a resolution of 32 x 32 pixels. The data is organised into 50,000 training samples and 10,000 test samples. CIFAR-10 is grouped into 10 classes with each class having exactly 6000 images. CIFAR-100 is grouped into 100 classes with each class having exactly 600 images. 

### Data loading Regime

- We load the CIFAR datasets from the [torchvision](https://pytorch.org/vision/stable/datasets.html#cifar) library which returns the data using the standard train-test split. We then perform the following transformations on the data prior to training models.

- We randomly split the test set into 5,000 validation samples and 5,000 test samples. This split is then preserved across all models trained. The indices of the 5000 samples selected from the torchvision test set to be used as the final test set for each dataset are included in this repo.

` CIFAR100_test_indexes.txt `
` CIFAR10_test_indexes.txt `

-  We apply mean and standard deviation normalisation to the data with the following parameters: 

| Dataset     | Mean | Standard Deviation |
|----------------|-------------|-------------|
|CIFAR-10        | [0.4914, 0.4822, 0.4465] | [0.2023, 0.1994, 0.2010] |
|CIFAR-100       | [0.5071, 0.4865, 0.4409] | [0.2009, 0.1984, 0.2023] |

-  The images are further transformed by adding 4 pixels white space padding to the edges followed by a random 32 x 32 crop of the enlarged image.

-  Finally we apply random horizontal flipping to our training data with probability 0.5.

# Model Parameters

## 1. CIFAR100 (Teacher & Baseline Student) and CIFAR10 (Teacher)

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            | 200 |
|Optimiser            | SGD |
|Learning Rate            | 0.1 initially, decayed to 0.01 and 0.001 at 80 and 120 epochs respectively|
|Momentum            | 0.9 |
|Weight Decay            | 5e-4 |
|Batch Size            | 64 |

Other optimisation parameters taken as default from [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html). The epoch with the highest classification accuracy on the validation dataset is selected as the final model weights.

## 2. CIFAR10 (Baseline Student)

All training parameters are identical to [section 1](#1-cifar100-teacher--baseline-student-and-cifar10-teacher) with the exception of:

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            | 100 |
|Learning Rate            | 0.1 initially, decayed to 0.01 and 0.001 at 40 and 60 epochs respectively|

## 3. KD Training Parameters

All training parameters from [section 1](#1-cifar100-teacher--baseline-student-and-cifar10-teacher) are used for KD student networks with CIFAR 100 or [section 2](#2-cifar10-baseline-student) for CIFAR10. The following two parameters specific to the KD loss function are also used in both cases.

| Parameter      | Value |
|----------------|-------------|
|*T*            | 8 |
|α            | 0.8 |

## 4. FitNet Training Parameters

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            | 500, early stopping after 100 epochs of no improvement on validation loss |
|Optimiser            | RMSprop |
|Learning Rate            | 0.005 |
|Batch Size            |  128|
|Layers Used            | layer selected approx one third of the way into each network  |

Other optimisation parameters taken as default from [torch.optim.RMSprop](https://pytorch.org/docs/stable/optim.html). The epoch with the lowest average loss on the validation dataset is selected as the final step 1 weights. We then take these step 1 weights as initialisation and train step 2 according to [section 1](#1-cifar100-teacher--baseline-student-and-cifar10-teacher) for CIFAR100 or [section 2](#2-cifar10-baseline-student) for CIFAR10. If we are combining FitNets with KD we train step 2 according to [section 3](#3-kd-training-parameters).

## 5. FSP Training Parameters

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            | 164 |
|Optimiser            | SGD |
|Learning Rate            | 0.01 initially, decayed to 0.001 and 0.0001 at 82 and 123 epochs respectively|
|Momentum            | 0.9 |
|Weight Decay            | 1e-4 |
|Batch Size            | 256 |
|Layers Used            | Flows taken over the first third, middle third and final third of each network |

Other optimisation parameters taken as default from [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html). The epoch with the lowest average loss on the validation dataset is selected as the final step 1 weights. We then take these step 1 weights as initialisation and train step 2 according to [section 1](#1-cifar100-teacher--baseline-student-and-cifar10-teacher) for CIFAR100 or [section 2](#2-cifar10-baseline-student) for CIFAR10. If we are combining FSP with KD we train step 2 according to [section 3](#3-kd-training-parameters).

## 6. PKT Training Parameters

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            | 500, early stopping after 100 epochs of no improvement on validation loss |
|Optimiser            | Adam |
|Learning Rate            | 1e-4 |
|Batch Size            | 128 |
|Layers Used            | Final convolutional layer |

Other optimisation parameters taken as default from [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html). The epoch with the lowest average loss on the validation dataset is selected as the final step 1 weights. We then take these step 1 weights as initialisation and train step 2 according to [section 1](#1-cifar100-teacher--baseline-student-and-cifar10-teacher) for CIFAR100 or [section 2](#2-cifar10-baseline-student) for CIFAR10. If we are combining PKT with KD we train step 2 according to [section 3](#3-kd-training-parameters).

## 7. MMD Training Parameters

| Parameter      | Value |
|----------------|-------------|
|**All Kernels**             ||
|Layers Used           | Final convolutional layer |
|**Linear Kernel**              ||
|β (CIFAR10)            | 6.0853e-3 |
|β (CIFAR100)           | 1.7122e-3 |
|**Polynomial Kernel**              ||
|*d*            | 2 |
|*c*            | 0 |
|β (CIFAR10)            | 7.7375e-20 |
|β (CIFAR100)            | 9.5243e-18|
|**Gaussian Kernel**              ||
|Kernel Mul           | 2 |
|N Kernels            | 5 |
|β (CIFAR10)            | 17.03 |
|β (CIFAR100)            | 26.5041|

The parameter β scales the MMD loss to an appropriate magnitude relevant to the cross-entropy label loss. This scaling was necessary to facilitate functional gradient descent. We calculated β prior to training and kept it constant throughout. The MMD loss is multiplied by β before being added to the label loss. When we train MMD with KD we switch out our cross-entropy label loss for the KD joint loss to form a triplet loss. We use the same KD parameters as in [section 3](#3-kd-training-parameters). All MMD parameters are the same when used with KD with the exception of one; we set β = 5.3008 for the Gaussian Kernel with KD on CIFAR100. We take all other base training parameters for the MMD models from [section 1](#1-cifar100-teacher--baseline-student-and-cifar10-teacher) for CIFAR100 or [section 2](#2-cifar10-baseline-student) for CIFAR10. 

## 8. CRD Training Parameters (CIFAR100)

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            | 240 |
|Optimiser            | SGD |
|Learning Rate            | 0.05 initially, multiplied by 0.1 at 150, 180 and 210 epochs|
|Momentum            | 0.9 |
|Weight Decay            | 5e-4 |
|Batch size            | 64 |
|Layers Used            | Final convolutional layer |
|β            | 0.8|
|*nce<sub>k<sub>*           | 4096 |
|*nce<sub>t<sub>*            | 0.07 |
|*nce<sub>m<sub>*            | 0.5 |

The parameter β scales the CRD loss to an appropriate magnitude before being added to the label loss. *nce<sub>k<sub>*, *nce<sub>t<sub>* and *nce<sub>m<sub>*  are CRD specific parameters with default values taken as per original paper. When we train CRD with KD we add to our β scaled CRD loss and label loss the KD soft loss to form a triplet loss. We use temperature *T*=8 as per [section 3](#3-kd-training-parameters).

## 9. CRD Training Parameters (CIFAR10)

| Parameter      | Value |
|----------------|-------------|
|Training Epochs            |180|
|Learning Rate            |0.01 initially, multiplied by 0.1 at 112, 135 and 157 epochs|

All other parameters the same as [section 8](#8-crd-training-parameters-cifar100).
