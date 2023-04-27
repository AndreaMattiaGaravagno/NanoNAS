import subprocess
import os
os.system('pip install pycocotools')
from pycocotools.coco import COCO

print("\nPlease be patient, it will take some time...\n")

os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip')
os.system('wget http://images.cocodataset.org/zips/train2014.zip')
os.system('wget http://images.cocodataset.org/zips/val2014.zip')
os.system('unzip annotations_trainval2014 -d data')
os.system('rm annotations_trainval2014.zip')
os.system('unzip train2014 -d data')
os.system('rm train2014.zip')
os.system('unzip val2014 -d data')
os.system('rm val2014.zip')
os.system('mkdir -p visual_wake_words/maxi_train/human')
os.system('mkdir -p visual_wake_words/maxi_train/no_human')
os.system('mkdir -p visual_wake_words/mini_val/human')
os.system('mkdir -p visual_wake_words/mini_val/no_human')
os.system('wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_minival_ids.txt')

with open('mscoco_minival_ids.txt', 'r') as f :
    minivalIds = list(map(int, f.read().splitlines()))

os.system('rm mscoco_minival_ids.txt')

#The label 1 is assigned as long as it has at least one bounding box
#corresponding to the object of interest (e.g. person) with the box area greater
#than 0.5% of the image area. - from https://arxiv.org/pdf/1906.05721.pdf
#The label 0 is assigned otherwise

#1 is person
def filter_ids(ids, foreground_class_of_interest_id=1, small_object_area_threshold=0.005) :
  filtered_ids = list()

  for id in ids :
    imAnn = coco.loadImgs(id)[0]
    Anns = coco.loadAnns(coco.getAnnIds(id))

    image_area = imAnn['height'] * imAnn['width']

    for annotation in Anns:
      normalized_object_area = annotation['area'] / image_area
      category_id = int(annotation['category_id'])
      # Filter valid bounding boxes
      if category_id == foreground_class_of_interest_id and \
          normalized_object_area > small_object_area_threshold:
        #if one is found, then the image is good
        #so you can append its index to the filtered ones
        filtered_ids.append(id)
        #and stop the search
        break
  discarded_ids = list(set(ids) - set(filtered_ids))

  return filtered_ids, discarded_ids

#Extracting images from COCO validation set 2014
pathToInstances = './data/annotations/instances_val2014.json'
coco = COCO(pathToInstances)

#get all images ids
allIds = coco.getImgIds()
img_count_val = len(allIds)
humanIds = coco.getImgIds(catIds=coco.getCatIds(['person']))
nonHumanIds = list(set(allIds) - set(humanIds))

humanIdsMaxtrain = list(set(humanIds) - set(minivalIds))
nonHumanIdsMaxtrain = list(set(nonHumanIds) - set(minivalIds))

humanIdsMinival = list(set(humanIds) - set(humanIdsMaxtrain))
nonHumanIdsMinival = list(set(nonHumanIds) - set(nonHumanIdsMaxtrain))

print("The following numbers should be equal")
print(f"{img_count_val}")
print(f"{len(humanIds) + len(nonHumanIds)}")

humanIdsMaxtrain, discardedHumanIdsMaxtrain = filter_ids(humanIdsMaxtrain)
humanIdsMinival, discardedHumanIdsMinival = filter_ids(humanIdsMinival)

nonHumanIdsMaxtrain = list(set(nonHumanIdsMaxtrain) | set(discardedHumanIdsMaxtrain))
nonHumanIdsMinival = list(set(nonHumanIdsMinival) | set(discardedHumanIdsMinival))

print(f"{len(humanIdsMaxtrain) + len(humanIdsMinival) + len(nonHumanIdsMaxtrain) + len(nonHumanIdsMinival)}")

#load images metadata
imagesWithHumansTraining = coco.loadImgs(humanIdsMaxtrain)
imagesWithoutHumansTraining = coco.loadImgs(nonHumanIdsMaxtrain)

imagesWithHumansValidation = coco.loadImgs(humanIdsMinival)
imagesWithoutHumansValidation = coco.loadImgs(nonHumanIdsMinival)

#write images' names onto a textual file
sourceFile = open('training_humans_image_list.txt', 'w')
for im in imagesWithHumansTraining :
  print('./data/val2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('training_no_humans_image_list.txt', 'w')
for im in imagesWithoutHumansTraining :
  print('./data/val2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('validation_humans_image_list.txt', 'w')
for im in imagesWithHumansValidation :
  print('./data/val2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('validation_no_humans_image_list.txt', 'w')
for im in imagesWithoutHumansValidation :
   print('./data/val2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

#move images from COCO to the new dataset using the textual files
print('Moving COCO 2014 validation images...')
os.system('cat training_humans_image_list.txt | xargs -I % mv % visual_wake_words/maxi_train/human')
os.system('cat training_no_humans_image_list.txt | xargs -I % mv % visual_wake_words/maxi_train/no_human')
os.system('cat validation_humans_image_list.txt | xargs -I % mv % visual_wake_words/mini_val/human')
os.system('cat validation_no_humans_image_list.txt | xargs -I % mv % visual_wake_words/mini_val/no_human')

os.system('rm training_humans_image_list.txt')
os.system('rm training_no_humans_image_list.txt')
os.system('rm validation_humans_image_list.txt')
os.system('rm validation_no_humans_image_list.txt')

#Extracting images from COCO training set 2014
pathToInstances = './data/annotations/instances_train2014.json'
coco = COCO(pathToInstances)

#get all images ids
allIds = coco.getImgIds()
img_count_train = len(allIds)
humanIds = coco.getImgIds(catIds=coco.getCatIds(['person']))
nonHumanIds = list(set(allIds) - set(humanIds))

humanIdsMaxtrain = list(set(humanIds) - set(minivalIds))
nonHumanIdsMaxtrain = list(set(nonHumanIds) - set(minivalIds))

humanIdsMinival = list(set(humanIds) - set(humanIdsMaxtrain))
nonHumanIdsMinival = list(set(nonHumanIds) - set(nonHumanIdsMaxtrain))

print("The following numbers should be equal")
print(f"{img_count_train}")
print(f"{len(humanIds) + len(nonHumanIds)}")

humanIdsMaxtrain, discardedHumanIdsMaxtrain = filter_ids(humanIdsMaxtrain)
humanIdsMinival, discardedHumanIdsMinival = filter_ids(humanIdsMinival)

nonHumanIdsMaxtrain = list(set(nonHumanIdsMaxtrain) | set(discardedHumanIdsMaxtrain))
nonHumanIdsMinival = list(set(nonHumanIdsMinival) | set(discardedHumanIdsMinival))

print(f"{len(humanIdsMaxtrain) + len(humanIdsMinival) + len(nonHumanIdsMaxtrain) + len(nonHumanIdsMinival)}")

#load images metadata
imagesWithHumansTraining = coco.loadImgs(humanIdsMaxtrain)
imagesWithoutHumansTraining = coco.loadImgs(nonHumanIdsMaxtrain)

imagesWithHumansValidation = coco.loadImgs(humanIdsMinival)
imagesWithoutHumansValidation = coco.loadImgs(nonHumanIdsMinival)

#write images' names onto a textual file
sourceFile = open('training_humans_image_list.txt', 'w')
for im in imagesWithHumansTraining :
  print('./data/train2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('training_no_humans_image_list.txt', 'w')
for im in imagesWithoutHumansTraining :
  print('./data/train2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('validation_humans_image_list.txt', 'w')
for im in imagesWithHumansValidation :
  print('./data/train2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('validation_no_humans_image_list.txt', 'w')
for im in imagesWithoutHumansValidation :
   print('./data/train2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

#move images from COCO to the new dataset using the textual files
print('Moving COCO 2014 training images...')
os.system('cat training_humans_image_list.txt | xargs -I % mv % visual_wake_words/maxi_train/human')
os.system('cat training_no_humans_image_list.txt | xargs -I % mv % visual_wake_words/maxi_train/no_human')
os.system('cat validation_humans_image_list.txt | xargs -I % mv % visual_wake_words/mini_val/human')
os.system('cat validation_no_humans_image_list.txt | xargs -I % mv % visual_wake_words/mini_val/no_human')

os.system('rm training_humans_image_list.txt')
os.system('rm training_no_humans_image_list.txt')
os.system('rm validation_humans_image_list.txt')
os.system('rm validation_no_humans_image_list.txt')

#let's compute the percentage of images labeled as humans inside the COCO Maxitrain set
p = subprocess.Popen(
    'ls visual_wake_words/maxi_train/no_human | wc -l',
    stdout=subprocess.PIPE, shell=True)
output, error = p.communicate()
maxitrain_no_humans_count = int(output)

p = subprocess.Popen(
    'ls visual_wake_words/maxi_train/human | wc -l',
    stdout=subprocess.PIPE, shell=True)
output, error = p.communicate()
maxitrain_humans_count = int(output)

#if it's 47%, respects the percentage reported in the paper about the visual wake word dataset
print('Percentage \'large\' humans (i.e. bounding box area > 0.5%) inside the maxitrain set: ' + str(round(100 * maxitrain_humans_count / (maxitrain_humans_count + maxitrain_no_humans_count),2)) + '%')

#let's compute the percentage of images labaled as humans inside the COCO MiniVal set
p = subprocess.Popen(
    'ls visual_wake_words/mini_val/no_human | wc -l',
    stdout=subprocess.PIPE, shell=True)
output, error = p.communicate()
minival_no_humans_count = int(output)

p = subprocess.Popen(
    'ls visual_wake_words/mini_val/human | wc -l',
    stdout=subprocess.PIPE, shell=True)
output, error = p.communicate()
minival_humans_count = int(output)

#if it's 47%, respects the percentage reported in the paper about the visual wake word dataset
print('Percentage \'large\' humans (i.e. bounding box area > 0.5%) inside the maxitrain set: ' + str(round(100 * minival_humans_count / (minival_humans_count + minival_no_humans_count),2)) + '%')
print('Both percentages should be around 47%')

print("The following numbers should be equal")
print(f"{img_count_train + img_count_val}")
print(f"{minival_humans_count + minival_no_humans_count + maxitrain_humans_count + maxitrain_no_humans_count}")

os.system('mv visual_wake_words/maxi_train/human/ visual_wake_words/maxi_train/1/')
os.system('mv visual_wake_words/maxi_train/no_human/ visual_wake_words/maxi_train/0/')
os.system('mv visual_wake_words/mini_val/human/ visual_wake_words/mini_val/1/')
os.system('mv visual_wake_words/mini_val/no_human/ visual_wake_words/mini_val/0/')

os.system('rm -rf data')

print(f"\nVisual Wake Word Dataset saved in: {os.path.abspath('./visual_wake_words/')}\n")
