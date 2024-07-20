## How to

Here, the major steps in developement phases, which would also apply, if one would introduce more data from the dataset provided by [Kaggele Competition: Lumbar Spine Gegenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data?select=train_images) are described.



### Examination of Data

Understanding the structure of the directories is crucial. Go to the directory `./helper_scripts`. In order to view dicom images, copy a desired path for a series into the helper script `show_dicom.py` (line 22). Then,

     python3.9 ./show_dicom.py


also, clean the provided .csv to only hold the actual available files

     python3.9 ./resize_csv_to_available_data.py

one might want to learn about the data, how many images, series (mri-scans) and studys (patients) are present:

     python3.9 ./count_images_per_series.py

and

     python3.9 ./count_series.py


### Setup (alternatively execute run_me.sh)

If not done already, clone this repository

     git clone --depth 1 https://github.com/warneat/ai-kimmerle (or gitlab...)

If present, remove the `venv` and create your own for your system

     rm -rf venv

Check if python3.9 is available with `python3.9 --version`. If not, install it. Create a new virtual environment and activate (development was in version 3.98)

     python3.9 -m venv venv
     source venv/bin/activate

Install requirements

     python3.9 -m pip install --upgrade pip setuptools
     python3.9 -m pip install -r requirements.txt

Clean and merge different labelfiles, also cleans to actual available data. This will create the final labelfiles

     python3.9 ./clean_csv.py

### Execute the Main Script

Execute the main script with

     python3.9 ./ai-kimmerle.py

The model and plots will be saved in `./saved_models_and_plots`