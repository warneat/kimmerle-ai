#!/usr/bin/env python3
import os
import pydicom
import matplotlib.pyplot as plt

def display_dicom(file_path):
    dicom_file = pydicom.dcmread(file_path)
    pixel_array = dicom_file.pixel_array
    plt.imshow(pixel_array, cmap=plt.cm.gray)
    plt.show()

def list_and_display_dicoms(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                print(f"Displaying {file_path}")
                display_dicom(file_path)


# Example directory containing DICOM files
directory = './small_set/train_images/1084486898/3897408956'

# List and display all DICOM files in the directory
list_and_display_dicoms(directory)
