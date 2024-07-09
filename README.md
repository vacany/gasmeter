## How to Start

[x] Make sure you have conda installed

```console
source install.sh
```

## How to Run

##### To run it on different image folder

```console
python main.py --image_folder ${PATH_TO_IMAGE_FOLDER}
```

##### To see printed values in rows with gradual increase

```console
python main.py --show_image_estimates 0
```

##### To see intermediate results

```console
python main.py --show_image_estimates 1
```

##### To try the application from scratch with cropping of value on gas meeters, remove the predefined box values:

```console
rm polygon.npy
```

##### To see format of Unfinished web app with uploading the images funcionality
```console
python webapp.py
```