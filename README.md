# Introduction
In this project, I developed a neural network model to classify handwritten digits using the MNIST dataset. The MNIST dataset is a benchmark dataset consisting of 60,000 training images and 10,000 test images of digits from 0 to 9, each represented as a 28x28 pixel grayscale image. The primary objective was to build a model capable of accurately predicting the digit in each image.

# Data Preprocessing
## Data Loading and Cleaning
The dataset was loaded and split into training and testing sets. Each image was flattened from a 28x28 matrix into a 784-dimensional vector to facilitate processing by the neural network.

## Normalization and Reshaping
The pixel values were normalized by dividing by 255, ensuring that they ranged between 0 and 1. This normalization step is crucial for improving the training process's efficiency and stability.

# Model Architecture
The neural network architecture consisted of multiple layers, including:

* Input Layer: 784 neurons (one for each pixel in the image).
* Hidden Layers: Three hidden layers with the following configuration:
  - First hidden layer: 128 neurons, ReLU activation function.
  - Second hidden layer: 64 neurons, ReLU activation function.
  - Third hidden layer: 32 neurons, ReLU activation function.
* Output Layer: 10 neurons (one for each digit class), with a softmax activation function to output probabilities for each class.
The choice of architecture, particularly the use of multiple hidden layers with ReLU activation, was aimed at capturing complex patterns in the data.

# Training the Model
## Forward Propagation
<ol>
  <li>Input Layer to First Hidden Layer:
    <ul>
      <li>z<sub>1</sub> = W<sub>1</sub> * X + b<sub>1</sub></li>
      <li>a<sub>1</sub> = σ(z<sub>1</sub>)</li>
    </ul>
  </li>
  <li>First Hidden Layer to Second Hidden Layer:
    <ul>
      <li>z<sub>2</sub> = W<sub>2</sub> * a<sub>1</sub> + b<sub>2</sub></li>
      <li>a<sub>2</sub> = σ(z<sub>2</sub>)</li>
    </ul>
  </li>
  <li>Second Hidden Layer to Output Layer:
    <ul>
      <li>z<sub>3</sub> = W<sub>3</sub> * a<sub>2</sub> + b<sub>3</sub></li>
      <li>a<sub>3</sub> = σ(z<sub>3</sub>)</li>
    </ul>
  </li>
</ol>
Where 𝜎 is the sigmoid activation function.

    def feed_forward(x, w1, w2, w3, b1, b2, b3):
        z1 = np.dot(w1, x.T) + b1
        a1 = sigmoid(z1)
    
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
    
        z3 = np.dot(w3, a2) + b3
        a3 = sigmoid(z3)
    
        return z1, a1, z2, a2, z3, a3

## Backward Propogation
<h3>Backward Propagation</h3>
<ol>
  <li>Output Layer to Second Hidden Layer:
    <ul>
      <li>δz<sub>3</sub> = a<sub>3</sub> - Y</li>
      <li>δW<sub>3</sub> = (1/m) * δz<sub>3</sub> * a<sub>2</sub><sup>T</sup></li>
      <li>δb<sub>3</sub> = (1/m) * Σδz<sub>3</sub></li>
    </ul>
  </li>
  <li>Second Hidden Layer to First Hidden Layer:
    <ul>
      <li>δz<sub>2</sub> = W<sub>3</sub><sup>T</sup> * δz<sub>3</sub> * σ'(z<sub>2</sub>)</li>
      <li>δW<sub>2</sub> = (1/m) * δz<sub>2</sub> * a<sub>1</sub><sup>T</sup></li>
      <li>δb<sub>2</sub> = (1/m) * Σδz<sub>2</sub></li>
    </ul>
  </li>
  <li>First Hidden Layer to Input Layer:
    <ul>
      <li>δz<sub>1</sub> = W<sub>2</sub><sup>T</sup> * δz<sub>2</sub> * σ'(z<sub>1</sub>)</li>
      <li>δW<sub>1</sub> = (1/m) * δz<sub>1</sub> * X<sup>T</sup></li>
      <li>δb<sub>1</sub> = (1/m) * Σδz<sub>1</sub></li>
    </ul>
  </li>
</ol>

    def back_propagation(z1, a1, z2, a2, a3, w2, w3, x, y):
        m = y.size
        one_hot_y = one_hot(y, num_classes=10)

        dz3 = a3 - one_hot_y
        a2 = a2.T
        dw3 = 1/m * np.dot(dz3, a2)
        db3 = 1/m * np.sum(dz3)
    
        w3 = w3.T
        dz2 = np.dot(w3, dz3) * derivative_sigmoid(z2)
        a1 = a1.T
        dw2 = 1/m * np.dot(dz2, a1)
        db2 = 1/m * np.sum(dz2)
    
        w2 = w2.T
        dz1 = np.dot(w2, dz2) * derivative_sigmoid(z1)
        dw1 = 1/m * np.dot(dz1, x)
        db1 = 1/m * np.sum(dz1)
    
        return dw1, db1, dw2, db2, dw3, db3

    def update_parameter(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
        w1 = w1 - alpha*dw1
        b1 = b1 - alpha*db1

        w2 = w2 - alpha*dw2
        b2 = b2 - alpha*db2
    
        w3 = w3 - alpha*dw3
        b3 = b3 - alpha*db3
    
        return w1, b1, w2, b2, w3, b3

## Mini-Batch Gradient Descent

    def mini_batch_gradient_descent(x_train, y_train, batch_size, epochs):
        w1, w2, w3, b1, b2, b3 = parameters()
        alpha = 0.01
        x_train = x_train.T
        m = x_train.shape[0]

        for epoch in range(epochs):
            for batch_start in range(0, m, batch_size):
                batch_end = batch_start + batch_size
                x_batch = x_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
    
                # Forward feed
                z1, a1, z2, a2, z3, a3 = feed_forward(x_batch, w1, w2, w3, b1, b2, b3)
    
                # Backward propagation
                dw1, db1, dw2, db2, dw3, db3 = back_propagation(z1, a1, z2, a2, a3, w2, w3, x_batch, y_batch)
    
                # Update parameters
                w1, b1, w2, b2, w3, b3 = update_parameter(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
    
            if epoch <= 100:
                print("Iteration: ", epoch)
                predictions = get_predictions(a3)
                print(get_accuracy(predictions, y_batch))
    
        return w1, b1, w2, b2, w3, b3

    

# Evaluation
The model achieved an accuracy of 88.20% on the test set. The accuracy was calculated by comparing the predicted labels with the true labels in the test set.

# Error Analysis
##Misclassified Images
* The model's performance was further analyzed by examining misclassified images.
* Out of 10,000 test images, 1,180 were misclassified. 
* The misclassifications were visualized to understand common errors and possible reasons, such as:
    -  Ambiguous or poorly written digits.
    -  Similar shapes between different digits (e.g., '1' and '7').
  
# Conclusion
* The neural network successfully classified handwritten digits with an accuracy of 88.20% on the test set. 
* While the model demonstrated strong performance, further improvements could be made, such as:
  - Fine-tuning the model architecture.
  - Implementing more advanced techniques like dropout or data augmentation.
  - Exploring different optimization algorithms.
