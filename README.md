# DeepFake-Detection
Deepfake detection application was developed to reveal whether deepfake videos are real or fake. For this purpose, Selim EfficientNet B7  and Conv. Cross ViT EfficientNet B0 [https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection] models were used. In addition to this feature, there is also a training tab for user-specific training of the Conv. Cross ViT EfficientNet B0 pretrained model.
##Deepfake Detection Inference Screen
The user first uploads the video they want to test. Then, the model or models to be run are selected. As a result, the model or models make inferences. The inference result with the highest accuracy is presented to the user whether the video is real or fake. In addition, the percentage of reliability of the result of the model is presented to the user.

![image](https://github.com/Efekanw/DeepFake-Detection/assets/56073720/0c511283-0887-4932-be68-3f2fa3b07471)
## Deepfake Detection Training Screen
The hyperparameters required for training the Conv. Cross ViT EfficientNet B0 model are provided by the user. Training, testing and validation datasets are also loaded to be used in training and testing the model. The accuracy, f1 score, loss, confusion matrix and roc curve obtained as a result of training the model are presented to the user.

![image](https://github.com/Efekanw/DeepFake-Detection/assets/56073720/1f54d8d4-7f78-451d-af7f-8b4d421e2597)
## Model Comparison
![image](https://github.com/Efekanw/DeepFake-Detection/assets/56073720/19a240c8-eaf3-4ce7-ab0b-09e5d0cfa62d)
### Selim EfficientNet B7
![image](https://github.com/Efekanw/DeepFake-Detection/assets/56073720/7b935708-1ffa-4b15-be57-f5f6ea84bee6)
### Conv. Cross ViT EfficientNet B0
![image](https://github.com/Efekanw/DeepFake-Detection/assets/56073720/71308e5c-5e0c-4f92-a928-c0623f3fde49)


