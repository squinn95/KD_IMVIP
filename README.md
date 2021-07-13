# Full Training Parameters

## 1. CIFAR100 (Teacher & Baseline Student) and CIFAR10 (Teacher)

Training Epochs: 200 Training Epochs:  
Optimiser: SGD  
Learning Rate: initially, decayed to 0.01 and 0.001 at 80 and 120 epochs
respectively  
Momentum:  
Weight Decay: 5e-4  
Batch size:

Other optimisation parameters taken as default from [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html). The epoch with the highest classification accuracy on the validation dataset is selected as the final model weights.

## 2. CIFAR10 (Baseline Student)

All training parameters are identical to section [1.1](#sec:default_params) with the exception of:

Training Epochs: 200 
Learning Rate: initially, decayed to 0.01 and 0.001 at 40 and 60 epochs respectively

## 3. KD Training Parameters

All training parameters from [section 1](#cifar100-teacher--baseline-student-and-cifar10-teacher) are used for KD student networks with CIFAR 100 or [section 2]() for CIFAR10. The following two parameters specific to the KD loss function are also used in both cases.

T:  
α:



## 4. FitNet Training Parameters

Training Epochs: 200 Training Epochs: , early stopping after 100 epochs of no improvement on validation loss  
Optimiser: RMSprop  
Learning Rate:  
Batch size:  
Layers Used: layer selected approx one third of the way into each network.

Other optimisation parameters taken as default from [torch.optim.RMSprop](https://pytorch.org/docs/stable/optim.html). The epoch with the lowest average loss on the validation dataset is selected as the final step 1 weights. We then take these step 1 weights as initialisation and train step 2 according to section [1.1](#sec:default_params) for CIFAR100 or [1.2](#sec:c10student) for CIFAR10. If we are combining FitNets with KD we train step 2 according to section [1.3](#sec:kd_params).

## 5. FSP Training Parameters

Training Epochs: 200 Training Epochs:  
Optimiser: SGD  
Learning Rate: initially, decayed to 0.001 and 0.0001 at 82 and 123 epochs respectively  
Momentum:  
Weight Decay: 1e-4  
Batch size:  
Layers Used: Flows taken over the first third, middle third and final third of each network.

Other optimisation parameters taken as default from [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html). The epoch with the lowest average loss on the validation dataset is selected as the final step 1 weights. We then take these step 1 weights as initialisation and train step 2 according to section [1.1](#sec:default_params) for CIFAR100 or [1.2](#sec:c10student) for CIFAR10. If we are combining FSP with KD we train step 2 according to section [1.3](#sec:kd_params).

## 6. PKT Training Parameters

Training Epochs: 200 Training Epochs: , early stopping after 100 epochs of no improvement on validation loss  
Optimiser: Adam  
Learning Rate: 1e-4  
Batch size:  
Layers Used: Final convolutional layer

Other optimisation parameters taken as default from [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html). The epoch with the lowest average loss on the validation dataset is selected as the final step 1 weights. We then take these step 1 weights as initialisation and train step 2 according to section [1.1](#sec:default_params) for CIFAR100 or [1.2](#sec:c10student) for CIFAR10. If we are combining PKT with KD we train step 2 according to section [1.3](#sec:kd_params).

## 7. MMD Training Parameters

**All Kernels:** 
Layers Used: Final convolutional layer

**Linear Kernel:**  

β (CIFAR10): 6.0853e-3  
β (CIFAR100): 1.7122e-3  
  
**Polynomial Kernel:**  
*d*:  
*c*:  
β (CIFAR10): 7.7375e-20  
β (CIFAR100): 9.5243e-18  
  
**Gaussian Kernel:**  
Kernel Mul:  
N Kernels:  
β (CIFAR10):  
β (CIFAR100):

The parameter β scales the MMD loss to an appropriate magnitude relevant to the cross-entropy label loss. This scaling was necessary to facilitate functional gradient descent. We calculated β prior to training and kept it constant throughout. The MMD loss is multiplied by β before being added to the label loss. When we train MMD with KD we switch out our cross-entropy label loss for the KD joint loss to form a triplet loss. We use the same KD parameters as in section [1.3](#sec:kd_params). All MMD parameters are the same when used with KD with the exception of one; we set β = 5.3008 for the Gaussian Kernel with KD on CIFAR100. We take all other base training parameters for the MMD models from [1.1](#sec:default_params) for CIFAR100 or [1.2](#sec:c10student) for CIFAR10. 

## 8. CRD Training Parameters (CIFAR100)

The following parameters are used for CRD joint-loss training for the CIFAR-100 dataset. 

Training Epochs: 240
Optimiser: SGD  
Learning Rate: initially, multiplied by 0.1 at 150, 180 and 210 epochs  
Momentum:  
Weight Decay: 5e-4  
Batch size:  
Layers Used: Final convolutional layer  
β:  
*nce_k*:  
*nce_t*:  
*nce_m*:

The parameter β scales the CRD loss to an appropriate magnitude before being added to the label loss. *nce_k*, *nce_t* and *nce_m* are CRD specific parameters with default values taken as per original paper. When we train CRD with KD we add to our β scaled CRD loss and label loss the KD soft loss to form a triplet loss. We use temperature T=8 as per section [1.3](#sec:kd_params).


## 9. CRD Training Parameters (CIFAR10)

All other parameters the same as [1.8](#sec:crd_default).

Training Epochs: 240 Training Epochs:  
Learning Rate: initially, multiplied by 0.1 at 112, 135 and 157 epochs.
