from argparse import ArgumentParser, BooleanOptionalAction
from ultralytics import YOLO
import glob
import random 
import yaml
from pathlib import Path
from tqdm import tqdm
from data import *
import shutil



# Build image/label directories 
def build_yolo_dirs(data_directories, n_samples, n_fft, class_list):
    labels = set()
    image_dirs = []
    label_dirs = []
    if not isinstance(data_directories, list):
        data_directories = [data_directories]
    print(f"{data_directories=}")
    for data_directory in data_directories:
        print(f"{data_directory=}")
        for f in tqdm(glob.glob(str(Path(data_directory,"*-meta")))):
            d = Data(f)
            d.generate_spectrograms(n_samples, n_fft, cmap_str="turbo", overwrite=False)
            yolo_label_outdir = str(Path(data_directory,"yolo","labels"))
            yolo_image_outdir = str(Path(data_directory,"yolo","images"))
            image_dirs.append(yolo_image_outdir)
            label_dirs.append(yolo_label_outdir)
            d.export_yolo(yolo_label_outdir, image_outdir=yolo_image_outdir, yolo_class_list=class_list)
    image_dirs = list(set(image_dirs))
    label_dirs = list(set(label_dirs))
    return image_dirs, label_dirs



def train_spec(
    train_dataset_path,
    val_dataset_path = None, 
    n_fft = 1024, 
    time_dim = 512,
    epochs = 50, 
    batch_size = -1, 
    class_list = None, 
    logs_dir = None, 
    output_dir = None,
):
    if logs_dir is None:
        logs_dir = datetime.now().strftime('spec_logs/%m_%d_%Y_%H_%M_%S')
    if output_dir is None:
        output_dir = "./"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    n_samples = n_fft * time_dim
    if not val_dataset_path: 
        image_dirs, label_dirs = build_yolo_dirs(train_dataset_path, n_samples, n_fft, class_list)
    
        image_files = []
        label_files = []
        for d in image_dirs: 
            image_files.extend(sorted(glob.glob(str(Path(d, "*")))))
        for d in label_dirs: 
            label_files.extend(sorted(glob.glob(str(Path(d, "*")))))
        assert len(image_files) == len(label_files)
        idxs = list(range(len(image_files)))
        random.shuffle(idxs)
        train_val_split = int(0.8*len(idxs))
        train_idxs = idxs[:train_val_split]
        val_idxs = idxs[train_val_split:]
    
        random_split_dir = Path("data", "random_split")
        if random_split_dir.is_dir():
            print(f"{random_split_dir} exists, overwriting...")
            shutil.rmtree(random_split_dir)
        
        random_split_train_dir = Path(random_split_dir, "train")
        random_split_train_dir.mkdir(parents=True, exist_ok=True)
        random_split_train_image_dir = Path(random_split_train_dir, "images")
        random_split_train_image_dir.mkdir(parents=True, exist_ok=True)
        random_split_train_label_dir = Path(random_split_train_dir, "labels")
        random_split_train_label_dir.mkdir(parents=True, exist_ok=True)
        
        random_split_val_dir = Path(random_split_dir, "val")
        random_split_val_dir.mkdir(parents=True, exist_ok=True)
        random_split_val_image_dir = Path(random_split_val_dir, "images")
        random_split_val_image_dir.mkdir(parents=True, exist_ok=True)
        random_split_val_label_dir = Path(random_split_val_dir, "labels")
        random_split_val_label_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in train_idxs:
            image_src = image_files[idx]
            label_src = label_files[idx]
            # Path(random_split_train_dir, "images").mkdir(parents=True, exist_ok=True)
            # Path(random_split_train_dir, "labels").mkdir(parents=True, exist_ok=True)
            image_dst = Path(random_split_train_image_dir, Path(image_src).name)
            label_dst = Path(random_split_train_label_dir, Path(label_src).name)
            shutil.copyfile(image_src, image_dst)
            shutil.copyfile(label_src, label_dst)
        for idx in val_idxs:
            image_src = image_files[idx]
            label_src = label_files[idx]
            # Path(random_split_val_dir, "images").mkdir(parents=True, exist_ok=True)
            # Path(random_split_val_dir, "labels").mkdir(parents=True, exist_ok=True)
            image_dst = Path(random_split_val_image_dir, Path(image_src).name)
            label_dst = Path(random_split_val_label_dir, Path(label_src).name)
            shutil.copyfile(image_src, image_dst)
            shutil.copyfile(label_src, label_dst)
        
        train_image_dirs = str(Path(Path.cwd(),random_split_train_image_dir))
        val_image_dirs = str(Path(Path.cwd(),random_split_val_image_dir))
    else: 
        train_image_dirs, train_label_dirs = build_yolo_dirs(train_dataset_path, n_samples, n_fft, class_list)
        
        val_image_dirs, val_label_dirs = build_yolo_dirs(val_dataset_path, n_samples, n_fft, class_list)
        
    
    # Write data.yaml
    yolo_yaml = {}
    yolo_yaml["train"] = train_image_dirs
    yolo_yaml["val"] = val_image_dirs
    yolo_yaml["names"] = class_list
    yolo_yaml["nc"] = len(class_list)
    with open('data.yaml', 'w') as outfile:
        yaml.dump(yolo_yaml, outfile, default_flow_style=False)
    print("Writing data.yaml")
    
    
    # YOLOv8
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='data.yaml', imgsz=640, batch=batch_size, epochs=epochs, project=str(output_dir), name=str(logs_dir))



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "train_dataset_path",
        type=str,
        nargs='+',
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        nargs='+',
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=1024,
        help="Length of FFT",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=512,
        help="Length of time dimension for spectrograms",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="Training example batch size",
    )
    parser.add_argument(
        "--class_list",
        nargs='+',
        type=str,
        help="List of classes to use",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="Path to write logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to write output",
    )

    return parser

if __name__ == "__main__":

    options = argument_parser().parse_args()
    
    # # Parameters
    # train_dataset_path = "data/gamutrf/gamutrf-sd-gr-ieee-wifi/test_offline/"
    # val_dataset_path = None
    
    # # Spectrogram parameters
    # n_fft = 1024
    # time_dim = 512
    # epochs = 50
    # n_samples = n_fft * time_dim
    # class_list = ["wifi", "anom_wifi"]
    # output_dir = "exp1"

    train_spec(
        train_dataset_path = options.train_dataset_path,
        val_dataset_path = options.val_dataset_path,
        n_fft = options.n_fft, 
        time_dim = options.time_dim,
        epochs = options.epochs, 
        batch_size = options.batch_size, 
        class_list = options.class_list, 
        logs_dir = options.logs_dir, 
        output_dir = options.output_dir,
    )
