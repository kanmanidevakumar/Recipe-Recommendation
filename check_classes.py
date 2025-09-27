import os

# Path to dataset folders
train_path = "dataset/train"
validation_path = "dataset/validation"

# Function to get class labels
def get_classes(dataset_path):
    return sorted(os.listdir(dataset_path))

if __name__ == "__main__":
    print("Training Classes:", get_classes(train_path))
    print("Validation Classes:", get_classes(validation_path))