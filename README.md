# Semantic Segmentation using Complex-Valued Neural Networks (CVNNs)

This project demonstrates the application of complex-valued neural networks for semantic segmentation tasks using datasets like **CamVid**. It features a custom **U-Net architecture** extended to handle complex-valued inputs and operations, allowing the network to leverage frequency-domain information for improved segmentation performance.

The core of this project is implemented in the `Semantic_Segmentation_CVNN.ipynb` notebook, which walks through data preprocessing, model definition, training, evaluation, and visualization. The dataset used is CamVid, structured into training, validation, and test folders, with corresponding label folders. A custom colormap is used to convert RGB label images to class indices.

The complex-valued U-Net is built using layers such as `ComplexConv2d`, `ComplexBatchNorm2d`, `ComplexReLU`, and `ComplexDropout`.  We have also used **ComplexAFF blocks** for attention-based refinement. During training, the model uses metrics like the F1 score to evaluate performance. Visualization steps overlay predicted masks on the original input images to qualitatively assess results.

To run this project, ensure you have the required dependencies installed. These include PyTorch, torchvision, numpy, matplotlib, opencv-python, scikit-learn, and PIL. 

After setting up the environment and dataset, open the notebook and run the cells sequentially. You can modify hyperparameters like learning rate, batch size, and number of epochs to suit your needs. The notebook provides clear outputs for loss curves, performance metrics, and segmented visual outputs.

Our results:
![image](https://github.com/user-attachments/assets/23f40714-0c6e-4531-84f4-cab188a9f537)


This implementation has achieved an macro **F1 score of approximately 0.7** on the CamVid dataset. Future directions include extending support to other datasets like Pascal VOC and Cityscapes, improving training speed and data augmentation, and experimenting with pre-trained backbones and more advanced complex-valued operations.

For any questions or collaboration inquiries, feel free to reach out. We hope this project serves as a useful starting point for exploring complex-valued neural networks in computer vision tasks.
