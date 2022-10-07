**Evolutionary convolutional neural network for efficient brain tumor segmentation and overall survival prediction**
---


Clone the repo: ```git clone https://github.com/FBehrad/Evo_conv.git ```


Segmentation 
---
1. Intall requirements.
```
pip install -r requirements.txt 
```
2. Put [BraTS 2018 training and validation dataset](https://ipp.cbica.upenn.edu/) next to config.yaml.
3. Run preprocessing.py.
4. Run augmentation.py.
5. Download model's weights from [this link](https://drive.google.com/file/d/1GFlbF2yiVdJeWddrSxRRELiTytrxeEAq/view?usp=sharing) and put it next to config.yaml.
6. Extract best_model.zip into best_model folder.
7. Change config.yaml to modify hyperparameters.
8. Run segmentation.py. (Training model)
9. Run prediction.py  (Create validation masks)
10. Upload the submission folder, created in step 9, into [this link](https://ipp.cbica.upenn.edu/) and get the segmentation results.
11. Run post_process.py and upload the new segmentation masks into the above link to get the new results. (Optional)

