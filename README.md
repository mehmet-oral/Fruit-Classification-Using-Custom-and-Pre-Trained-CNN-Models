Fruit Image Classification using a Custom Deep CNN

Project Overview
This project implements a complete deep learning pipeline for fruit image classification using a Convolutional Neural Network (CNN) built from scratch with NumPy. The model classifies fruit images into 24 different categories using only fundamental Python libraries without high-level deep learning frameworks.

Dataset Structure
The dataset is organized into two main directories:
- Training/ - Contains subfolders for each fruit class with training images
- Test/ - Contains subfolders for each fruit class with testing images

Each class has its own folder named after the fruit type (e.g., "apple", "banana", etc.).

Required Libraries
This implementation requires the following Python libraries:
- NumPy - For numerical computations and array operations
- PIL (Pillow) - For image loading and processing
- Matplotlib - For data visualization and plotting
- Seaborn - For enhanced visualization (confusion matrix)
- scikit-learn - For evaluation metrics (confusion matrix)
- os - For file system operations and path handling
- pickle - For model serialization and saving weights
- shutil - For file operations (copying dataset)
- psutil - For system monitoring (RAM checking)
- random - For randomization operations

Image Preprocessing
Images undergo the following preprocessing steps:
1. Conversion to grayscale
2. Resizing to 100×100 pixels
3. Normalization of pixel values to the range [0, 1]
4. Structured loading with corresponding labels

Model Architecture
The custom CNN implementation includes:

Convolutional Layers:
- First convolutional layer: 8 filters (3×3 kernel size)
- Second convolutional layer: 16 filters (3×3 kernel size)

Pooling Layers:
- Two max-pooling layers with 2×2 pooling size

Fully Connected Layer:
- Final classification layer with softmax activation

Activation Function:
- ReLU activation after each convolutional layer

Implementation Features

Model Operations:
- Forward propagation through all layers
- Backpropagation with weight updates
- Save/load model weights functionality
- Comprehensive evaluation metrics

Visualization Tools:
- Filter visualization for both convolutional layers
- Confusion matrix display
- Training progress monitoring

Training Configuration:
- Adjustable learning rate
- Configurable batch size
- Customizable number of epochs
- Weight persistence options

Usage Instructions

1. Data Preparation:
Mount Google Drive and copy the dataset to the runtime environment.

2. Model Initialization:
Initialize the CNN model with specified input shape, number of classes, and learning rate.

3. Training:
Train the model on the training data with specified epochs and batch size.

4. Evaluation:
Evaluate model performance on the test set to get predictions, loss, and accuracy.

5. Visualization:
Visualize learned filters from convolutional layers and plot confusion matrix.

6. Model Persistence:
Save and load model weights for future use.

Performance Considerations
- The implementation includes RAM monitoring for resource management
- Recommended to use a high-RAM runtime for optimal performance
- Batch processing minimizes memory usage during training

Outputs
The implementation provides:
- Training loss and accuracy metrics
- Test set performance evaluation
- Visualizations of learned features
- Confusion matrix for model assessment
- Persistent model weights for future use

Educational Value
This implementation demonstrates:
- Fundamental CNN operations without high-level abstractions
- Manual implementation of forward and backpropagation
- Weight update mechanisms using gradient descent
- Model evaluation and visualization techniques

Note
This custom implementation is designed for educational purposes to understand the underlying mechanics of convolutional neural networks. For production use, consider using optimized deep learning frameworks like TensorFlow or PyTorch.
