### Finding a dataset

dataset should be in my interest. i did not want to work with simple numbers etc only. Real world useful data
-> using DICOM Medical Dataset MRI Spine Scans for stenosis analysis from Kaggle. High personal interest in Image preprocessing.

### Looking at the data

First things first. Looking at the data, quality, what labels are available, how are they presented.

### Looking at the Data and Structure

with some medical background it was clear, which images could be more useful than others. There are many possibilities at when to put the images in the AI-Algorhithms, Since i could search for specific regions of interest on hard criteria beforehand or not.

### Finding oportunities for image selection

Tests with vertical lines and counting contrast changes not useful

### Making tests on image preprocessing like normalisation etc. 

Rotations and mirroring etc are not performed at the beginning, since the dataset did not show any unusual rotations in any case. Color curves could be performed so that nerve tissue is enhanced. This could bring might bring a benefit.

### Problems with installing runner

lrz does not seem to allow giving runner tokens. settings > CI/CD is not available.
Now Co-Owner of group and could install the container on my mac which will be transferred to a powerful computer. I decided to install a regular runner, workin on shell basis, not docker

### finding a lightweight pretrained network

Online Research gave me several Options for choosing a pretrained net. Densenet, resnet, unet... MONAI Framework seemed well documented and provides pretrained networks for medical purposes

### Pytorch vs keras vs monai

- monai: Pytorch based, pretrained on medical data -> less preprocessing


### Script to just load Data

- Loading data and merging to information from csv file.


### make problem less complex

Some given information could be ignored, such as condition coordinates,
The pretrained net is specific for medical purposes and should not need the series description information. This let´s me make the problem less complex.

### performance was really bad on level based. 

- idea to combine images from a study_id or series id (one spine)


### many problems on dimensions of tensors when combining images 

- to take advantage of images in series being connected

### many problems Dataloader

- stacked tensor not the right size
- removed multiprocessing, made learning unreliable

### only take 15

- after checking, what number of images are mostly present which is 15 i will use Augmentation to

### problems with dimensions. Monai provides different nets

- in densenet 212 not solveable
- now unet

### WORKING?

so what we basically do is

- taking 15 images which belong together
- combine them in a tensor
- make a batch out of them?
- send them in the unet

### hundreds of steps intrying using multiple slides 

- of series but could not encode the labels

### Back to learning with single images and it´s label.

- still had to create the label per image, since only labels for problematic cases are available

### Performance mixed:

- excellent performance on certain kinds of conditions
- bad performance on level, since axial slides are similar through spine (5 Levels, 5 conditions (foraminal and))

### Labels for all images

- labels are now available for every image after loading. Only the ones with probleatic conditions were provided.


### Contextual Info?

- Model performed bad in Level prediction.
- Idea is to give it some contextual information from the slides above and below but only looking at the central labels. first attempt with 3 in total, second attempt with 5. Mixed performance.

- Changing hyperparameters