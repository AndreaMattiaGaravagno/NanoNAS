from NanoNAS import NanoNAS

input_shape = (50,50,3)

#Each dataset must comply with the following structure
#main_directory/
#...class_a/
#......a_image_1.jpg
#......a_image_2.jpg
#...class_b/
#......b_image_1.jpg
#......b_image_2.jpg
path_to_training_set = '../../datasets/melanoma_cancer_dataset/train'
val_split = 0.3
path_to_test_set = '../../datasets/melanoma_cancer_dataset/test'

#whether or not to cache datasets in memory
#if the dataset cannot fit in the main memory, the application will crash
cache = True

#target: STM32L010RBT6
#75 CoreMark, 20 kiB RAM, 128 kiB Flash
ram_upper_bound = 20480
flash_upper_bound = 131072
MACC_upper_bound = 750000 #CoreMark * 1e4

nanoNAS = NanoNAS(ram_upper_bound, flash_upper_bound, MACC_upper_bound, path_to_training_set, val_split, cache, input_shape, save_path='./results')

#search
nanoNAS.search(save_search_history=False)

#train resulting architecture
nanoNAS.train(training_epochs=100, training_learning_rate=0.01, training_batch_size=128)

#apply uint8 post trainig quantization
nanoNAS.apply_uint8_post_training_quantization()

#evaluate post training quantization
nanoNAS.test_keras_model(path_to_test_set)
nanoNAS.test_tflite_model(path_to_test_set)
