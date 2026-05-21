from utils.file_utils import generate_dataset

if __name__ == "__main__":
    
    generate_dataset(
        image_folder="training_samples/nike",
        dataset_train_folder="dataset/train",
        dataset_eval_folder="dataset/eval",
        train_eval_split=0.5,
    )