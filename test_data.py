# Tests and helper scripts for data.py

from data import Data, labels_to_sigmf, yield_label_metadata


def test_spectrogram_generation(filename):
    d = Data(filename)
    n_fft = 1024
    time_dim = 512
    n_samples = n_fft * time_dim
    image_outdir = filename + "_images"
    d.generate_spectrograms(n_samples, n_fft, image_outdir=image_outdir)


def test_gamutrf_spectrogram_generation(filename):
    d = Data(filename)
    n_fft = 1024
    time_dim = 512
    n_samples = n_fft * time_dim
    image_outdir = filename + "_gamutrf_images"
    d.generate_gamutrf_spectrograms(n_samples, n_fft, image_outdir=image_outdir)


def test_import_labelme(directory):
    label_ext = ".json"
    label_type = "labelme"
    label_directory = directory + "png/"
    image_directory = directory + "png/"
    samples_directory = directory + "samples/"
    metadata_directory = directory + "metadata/"
    labels_to_sigmf(
        yield_label_metadata(
            label_ext,
            label_directory,
            image_directory,
            samples_directory,
            metadata_directory,
        ),
        label_type,
    )


def test_import_yolo(directory):
    label_ext = ".txt"
    label_type = "yolo"
    label_directory = directory + "YOLODataset/labels/train/"
    image_directory = directory + "YOLODataset/images/train/"
    samples_directory = directory + "samples/"
    metadata_directory = directory + "metadata/"
    yolo_dataset_yaml = directory + "YOLODataset/dataset.yaml"
    labels_to_sigmf(
        yield_label_metadata(
            label_ext,
            label_directory,
            image_directory,
            samples_directory,
            metadata_directory,
        ),
        label_type,
        yolo_dataset_yaml,
    )


def test_export_yolo(directory, filename):
    d = Data(filename)
    yolo_label_outdir = f"{directory}yolo/labels"
    yolo_image_outdir = f"{directory}yolo/images"
    d.export_yolo(yolo_label_outdir, image_outdir=yolo_image_outdir)


def test_export_labelme(directory, filename):
    d = Data(filename)
    labelme_outdir = f"{directory}labelme/"
    d.export_labelme(labelme_outdir, image_outdir=labelme_outdir)


def test_auto_label(filename):
    d = Data(filename)
    n_fft = 1024
    time_dim = 512
    n_samples = n_fft * time_dim
    d.generate_spectrograms(n_samples, n_fft, cmap_str="Greys", overwrite=True)

    d.auto_label_spectrograms("dji")


def test_regenerate(filename):
    test_auto_label()
    d = Data(filename)
    n_fft = 1024
    time_dim = 512
    n_samples = n_fft * time_dim
    d.generate_spectrograms(n_samples, n_fft, cmap_str="turbo", overwrite=True)

    label_outdir = filename + "_yolo_labels"
    image_outdir = filename + "_yolo_images"
    d.export_yolo(label_outdir=label_outdir, image_outdir=image_outdir)


def test_img_dimensions(filename):
    d = Data(filename)
    n_fft = 1024
    time_dim = 512
    n_samples = n_fft * time_dim
    image_outdir = filename + "_images"
    d.generate_spectrograms(
        n_samples, n_fft, cmap_str="Greys", image_outdir=image_outdir, overwrite=True
    )

    d.auto_label_spectrograms("dji")
    gamutrf_image_outdir = filename + "_gamutrf_images"
    d.generate_gamutrf_spectrograms(n_samples, n_fft, image_outdir=gamutrf_image_outdir)

    labelme_outdir = f"{filename}_labelme/"
    d.export_labelme(labelme_outdir, image_outdir=labelme_outdir)
