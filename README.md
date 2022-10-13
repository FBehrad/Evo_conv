**Evolutionary convolutional neural network for efficient brain tumor segmentation and overall survival prediction**
---
+ Read the paper [here](https://doi.org/10.1016/j.eswa.2022.118996).


---
Clone the repo: ```git clone https://github.com/FBehrad/Evo_conv.git ```

Segmentation 
---
1. Intall requirements.
```
pip install -r requirements.txt 
```
2. Put [BraTS 2018 training and validation dataset](https://ipp.cbica.upenn.edu/) in the main directory (next to config.yaml).
3. Run preprocessing.py.
4. Run augmentation.py.
5. Download model's weights from [this link](https://drive.google.com/file/d/1GFlbF2yiVdJeWddrSxRRELiTytrxeEAq/view?usp=sharing) and put it in the main directory.
6. Extract best_model.zip into best_model folder.
7. Change config.yaml to modify hyperparameters.
8. Run segmentation.py. (Training model)
9. Run prediction.py  (Create validation masks)
10. Upload the submission folder, created in step 9, into [this link](https://ipp.cbica.upenn.edu/) and get the segmentation results.
11. Put Stats_Validation_final.csv in the main directory.
12. Run post_process.py and upload the new segmentation masks into the above link to get the new results. (Optional)

Pruning 
---
We used [keras-surgeaon](https://github.com/BenWhetton/keras-surgeon) to prune our network. However, Keras-surgeon does not support group normalization, so we have changed its code slightly. The new version is available in the pruning directory.

1. Run pruning.py.
2. Run prediction.py.

OS 
---
1. Run preprocessing.py.

Citation 
---
If you found this code helpful, please consider citing:
```
@article{BEHRAD2022118996,
title = {Evolutionary convolutional neural network for efficient brain tumor segmentation and overall survival prediction},
journal = {Expert Systems with Applications},
pages = {118996},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.118996},
url = {https://www.sciencedirect.com/science/article/pii/S0957417422020140},
author = {Fatemeh Behrad and Mohammad {Saniee Abadeh}},
keywords = {Deep learning, Genetic algorithm, Network compression},
abstract = {The most common and aggressive malignant brain tumor in adults is glioma, which leads to short life expectancy. A reliable and efficient automatic segmentation method is beneficial for clinical practice. Deep neural networks have achieved great success in brain tumor segmentation recently. However, their computational complexity and storage costs hinder their deployment in real-time applications or on resource-constrained devices in clinics. Network pruning, which has attracted many researchers recently, alleviates this limitation by removing trivial parameters of the network. Network pruning is challenging in the medical field because pruning should not degrade the performance of models. As a result, it is vital to choose unimportant parts of networks correctly. In this paper, we employ the genetic algorithm to identify redundant filters of our U-Net-based network. We consider filter pruning a multiobjective optimization problem in which performance and inference time are optimized simultaneously. Then, we use our compressed network for brain tumor segmentation. Predicted segmentation masks are often used to predict patients' survival time. Although several studies have recently achieved good results, they require different feature engineering techniques to extract suitable features, which is difficult and time-consuming. To tackle this problem, we easily extract deep features from the endpoint of the encoder of our compact network and use them for survival prediction. Regarding the popularity of U-Net-based models for brain tumor segmentation, many researchers can employ our technique to predict the survival time without spending lots of time on feature engineering. The experimental results on the BraTS 2018 dataset demonstrate that filter pruning is a reliable technique to reduce the storage cost and accelerate the network during inference while maintaining performance. Furthermore, our survival time prediction technique achieves high efficiency compared to state-of-the-art methods. Preprocessed data, the full implementation of the project, and the trained networks are available at https://github.com/Fbehrad/Evo_conv.}
}
```
