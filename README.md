INSTRUCTIONS ON HOW TO RUN
___________________________________



CLASSIFYING

cd into the classification-code directory

You can test on a single image. Modify the image path as desired.

To classify into one of the 15 scenes:
-     python run.py --task 4 --data ../15_Scene --imagePath ../15_Scene/test/Office/image_0001.jpg

To classify into cartoon-or-not using custom "your" model:
-     python run.py --task 5 --data ../cartoon-or-not --imagePath ../cartoon-or-not/test/real/image_0001.jpg

To classify into cartoon-or-not using custom VGG model:
-     python run.py --task 6 --data ../cartoon-or-not --imagePath ../cartoon-or-not/test/cartoon/image_0001.jpg

* If wanting to retrain, you might want to adjust "hyperparameters_sc" based on the task. I have already trained and saved every model's weights *

But order to run/train the cartoon-or-not classification model enter the following into the terminal:

FOR OUR OWN CUSTOM MODEL: 
-     python run.py --task 2 --data ../cartoon-or-not
-     adjustments may need to be made to "hyperparameters_sc" as it is currently set up to work with the VGG aproach which had better results; please see comment in file.
FOR VGG BASED MODEL:
-     python run.py --task 3 --data ../cartoon-or-not 




APPLYING FILTERS

Cd into the filter-code directory

Running through the terminal: python3 main.py -t | --task <cartoon or edge or both> -q | --quantity <single or many> -i | --image <image or nested folder path>

Based on the input for "--task", you can apply either a "cartoon" or an "edge" filter. Or "both".

Based on the input for "--quantity", you can either apply a filter to a "single" image or to "many" images." For a single image, paste in the pathway to the image for the "--image" argument. For many images, paste in the pathway to a folder of folders of images for the "--image" argument.
