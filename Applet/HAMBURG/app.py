
# import all necessary libraries
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, cifar10, mnist 
from sklearn.model_selection import train_test_split

# set the page icon
st.set_page_config(page_icon="🧍‍♂️",)

# Function to load data for a given dataset which will give as an argumnet
def load_data(dataset_name):
    # Checking if the dataset name is Fashion MNIST
    if dataset_name == "Fashion MNIST":
        # Loading Fashion MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        # class names for different fashion items
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # Normalizing pixel values to be between 0 and 1
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    # Checking if the dataset name is CIFAR-10
    elif dataset_name == "CIFAR-10":
        # Loading CIFAR-10 dataset
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        # class names for different objects in CIFAR-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # Converting pixel values to float32 and normalizing them to be between 0 and 1
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        # One-hot encoding the target class (labels)
        num_classes = 10
        train_labels = to_categorical(train_labels, num_classes)
        test_labels = to_categorical(test_labels, num_classes)
    # if the dataset name is MNIST
    elif dataset_name == "MNIST":
        # Loading MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        # class names for different digits
        class_names = [str(i) for i in range(10)]
        # Reshaping images to have a single channel
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
        # One-hot encoding the target class (labels)
        num_classes = 10
        train_labels = to_categorical(train_labels, num_classes)
        test_labels = to_categorical(test_labels, num_classes)

    # Handling the case of an invalid dataset name
    else:
        raise ValueError("Invalid dataset name. Supported options are 'Fashion MNIST' and 'CIFAR-10'.")
    
    # Returning the loaded data
    return train_images, train_labels, test_images, test_labels, class_names

# Function to create a neural network model based on the given dataset and input shape (like it will be 28*28 or 32*32 )
def create_model(dataset_name, input_shape):
    # Checking if the dataset is Fashion MNIST
    if dataset_name == "Fashion MNIST":
        # sequential model for Fashion MNIST
        model = tf.keras.Sequential([
            # Flattening the input images
            tf.keras.layers.Flatten(input_shape=input_shape),
            # Adding a dense layer with 128 neurons and ReLU activation
            tf.keras.layers.Dense(128, activation='relu'),
            # Output layer with 10 neurons 
            tf.keras.layers.Dense(10)
        ])
        # Compiling the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
        model.compile(optimizer="adam",
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    # if the dataset is CIFAR-10
    if dataset_name == "CIFAR-10":
        # sequential model for CIFAR-10
        model = tf.keras.Sequential([
            # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and padding
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            # Max pooling layer
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Convolutional layer with 64 filters, 3x3 kernel, ReLU activation, and padding
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            # Max pooling layer
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Flattening the convolutional output
            tf.keras.layers.Flatten(),
            # Dense layer with 128 neurons and ReLU activation
            tf.keras.layers.Dense(128, activation='relu'),
            # Output layer with 10 neurons (classes) and softmax activation
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Compiling the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    # if the dataset is MNIST
    if dataset_name == "MNIST":
        # Sequential model for MNIST
        model = tf.keras.Sequential([
            # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and padding
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            # Max pooling layer
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Convolutional layer with 64 filters, 3x3 kernel, ReLU activation, and padding
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            # Max pooling layer
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Flattening the convolutional output
            tf.keras.layers.Flatten(),
            # Dense layer with 128 neurons and ReLU activation
            tf.keras.layers.Dense(128, activation='relu'),
            # Output layer with 10 neurons (classes) and softmax activation
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Compiling the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    
    # Returning the required model
    return model

# Function to train a given neural network model on the training data
def train_model(model, X_train, y_train, epochs=10):
    # Training the model and storing training history
    history = model.fit(X_train, y_train, epochs=epochs)
    # Returning the trained model, training loss, and training accuracy
    return model, history.history['loss'], history.history['accuracy']

# Function to evaluate a trained neural network model on the test data
def evaluate_model(model, X_test, y_test):
    # Evaluating the model on the test data and getting test loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # Returning the test accuracy
    return test_acc

# Function to make predictions using a trained model on a any image
def make_prediction(dataset_name, model, image, class_names, input_shape):
    # if the dataset is Fashion MNIST
    if dataset_name == "Fashion MNIST":
        # Convert the PIL image to a NumPy array and normalize pixel values
        image_array = np.array(image.resize(input_shape[:2])) / 255.0
        # Reshape the image array to match the model's input shape
        image_array = image_array.reshape((1,) + input_shape)
        # Make predictions using the trained model
        predictions = model.predict(image_array)
        # Get the predicted class name based on the highest probability
        predicted_class = class_names[np.argmax(predictions)]
    
    # if the dataset is CIFAR-10
    if dataset_name == "CIFAR-10":
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)
        # Resize the image using tf.image.resize and normalize pixel values
        image_array = tf.image.resize(image_array, (input_shape[0], input_shape[1])) / 255.0
        # Expand dimensions to match the model's input shape
        image_array = np.expand_dims(image_array, axis=0) 
        # Make predictions using the trained model
        predictions = model.predict(image_array)
        # Get the predicted class name based on the highest probability
        predicted_class = class_names[np.argmax(predictions)]

    # Checking if the dataset is MNIST
    if dataset_name == "MNIST":
        # Convert the PIL image to a NumPy array and normalize pixel values
        image_array = np.array(image.resize(input_shape[:2])) / 255.0
        # Reshape the image array to match the model's input shape
        image_array = image_array.reshape((1,) + input_shape)
        # Make predictions using the trained model
        predictions = model.predict(image_array)
        # Get the predicted class name based on the highest probability
        predicted_class = class_names[np.argmax(predictions)]

    # Return the predicted class name
    return predicted_class


# Function to save a trained model to a file based on the dataset name
def save_model(dataset_name, model, model_filename="model.h5"):
    # if the dataset is Fashion MNIST
    if dataset_name == "Fashion MNIST":
        # Save the model
        model.save(f"fashion_{model_filename}")
    
    # if the dataset is CIFAR-10
    if dataset_name == "CIFAR-10":
        # Save the model 
        model.save(f"cifar_{model_filename}")
    
    # if the dataset is MNIST
    if dataset_name == "MNIST":
        # Save the model 
        model.save(f"mnist_{model_filename}")


# Function to load a saved model from a file based on the dataset name
def load_saved_model(dataset_name, model_filename="model.h5"):
    # if the dataset is Fashion MNIST
    if dataset_name == "Fashion MNIST":
        # Load the saved model for Fashion MNIST
        model = tf.keras.models.load_model(f"fashion_{model_filename}")
    
    # Checking if the dataset is CIFAR-10
    if dataset_name == "CIFAR-10":
        # Load the saved model for CIFAR-10
        model = tf.keras.models.load_model(f"cifar_{model_filename}")

    # Checking if the dataset is MNIST
    if dataset_name == "MNIST":
        # Load the saved model for MNIST
        model = tf.keras.models.load_model(f"mnist_{model_filename}")
    
    # Return the loaded model
    return model



# accuracy and loss history
accuracy_history = []
loss_history = []


def main():
    st.sidebar.header("Select Action")
    action = st.sidebar.radio("Choose Action", ["Test Image", "Retrain Model", "Guide Lines"])
    dataset_name = None  

    # Guide Lines action : it will show the guides lines how to use application 
    if action == "Guide Lines":

        st.markdown("# User Guide for Your Web App")

        ## 1. Retrain the Model:
        st.markdown("### 1. Retrain the Model:")
        st.write("- Click on 'Retrain Model' in the sidebar.")
        st.write("- Choose the dataset you want to use (Fashion MNIST, CIFAR-10, or MNIST).")
        st.write("- Adjust retraining options like the number of epochs, training size, and click the 'Retrain Model' button.")
        st.write("- Watch the training progress and see accuracy and loss graphs.")

        ## 2. Test Image:
        st.markdown("### 2. Test Image:")
        st.write("- Click on 'Test Image' in the sidebar.")
        st.write("- Select the dataset you trained the model on.")
        st.write("- Upload an image using the file uploader.")
        st.write("- Click the 'Predict' button to see what the model thinks about the image.")
        st.write("- The predicted result will be displayed on the webpage.")

        ## 3. General Tips:
        st.markdown("### 3. General Tips:")
        st.write("- Make sure to wait for the model to finish training before expecting accurate predictions.")
        st.write("- For testing an image, choose an appropriate dataset for the trained model.")



    # Retrain Model action : here we will retrain the model
    if action == "Retrain Model":
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Fashion MNIST", "CIFAR-10", "MNIST"])
        st.title(f"Retrain the {dataset_name} Model")
        train_images, train_labels, test_images, test_labels, class_names = load_data(dataset_name)
        num_train_images = len(train_images)
        num_test_images = len(test_images)
        st.sidebar.subheader(f"{dataset_name} Dataset Information")
        st.sidebar.text(f"Number of Training Images: {num_train_images}")
        st.sidebar.text(f"Number of Test Images: {num_test_images}")
        st.sidebar.subheader("Retraining Parameters")
        retrain_epochs = st.sidebar.slider("Epochs for Retraining", min_value=1, max_value=50, value=10, step=1)
        train_size = st.sidebar.slider("Training Size", min_value=0.1, max_value=0.9, value=0.8, step=0.01)
        test_size = 1 - train_size
    
    
        # Add a button to trigger the retraining
        selected = st.sidebar.button("Retrain Model")
        if selected:
            train_images, train_labels, test_images, test_labels, class_names = load_data(dataset_name)
            X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=test_size, random_state=42)
    
            model_input_shape = X_train.shape[1:]
            model = create_model(dataset_name, model_input_shape)

            # Create a placeholder for displaying the training progress
            progress_placeholder = st.sidebar.empty()

            # Train the model and show the progress dynamically
            for epoch in range(retrain_epochs):
                model, loss, acc = train_model(model, X_train, y_train, epochs=1)
                # Update the placeholder with the current epoch
                progress_placeholder.text(f"Training Epoch: {epoch + 1}/{retrain_epochs}")
                # Append accuracy and loss to history
                accuracy_history.append(acc)
                loss_history.append(loss)

                # Display training progress on the main page
                st.text(f"The {epoch + 1} epoch out of {retrain_epochs}  has been completed, achieving an accuracy of {acc[0]:.4f}")
            
            # Show the accuracy dynamically
            test_acc = evaluate_model(model, X_test, y_test)
            st.sidebar.text(f'Test Accuracy: {test_acc:.4f}')

            # Save the model after training
            save_model(dataset_name, model)

            # graphs of accuracy and loss
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(range(1, retrain_epochs + 1), accuracy_history, color="green", marker="*", label='Accuracy')
            ax1.set(xlabel='Epoch', ylabel='Accuracy', title=f'{dataset_name} Model Training Accuracy At Each Epoch')
            ax1.grid()
            ax1.legend()

            ax2.plot(range(1, retrain_epochs + 1), loss_history, color="red", marker="o", label='Loss')
            ax2.set(xlabel='Epoch', ylabel='Loss', title=f'{dataset_name} Model Training Loss At Each Epoch')
            ax2.grid()
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)
    
    # test image action : we will upload new image and model will predict
    if action == "Test Image":
        st.title("Image Recognition App - Neural Network")
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Fashion MNIST", "CIFAR-10", "MNIST"])
        st.subheader(f"Upload an Image for {dataset_name}")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Display original image size
            st.write(f"Original Image Size: {image.size[0]} x {image.size[1]} pixels")
            st.image(image, caption=f"Uploaded Image", use_column_width=False)
            st.image(image, caption=f"Uploaded Image", use_column_width=True)
            train_images, train_labels, test_images, test_labels, class_names = load_data(dataset_name)
            model_input_shape = train_images.shape[1:]
            # Load the saved model
            model = load_saved_model(dataset_name)
            predict = st.button("Predict")
            if predict:
                # print the prediction on the webpage
                prediction = make_prediction(dataset_name, model, image, class_names, model_input_shape)
                st.markdown(f"<h2 style='text-align: center; color: green;'>Prediction: {prediction}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()