# NanoNAS
A small hardware-aware neural architecture search algorithm targeting low-end microcontrollers. Given its low search cost, it can be executed on laptops without a GPU.

# News
* **2023/04** NanoNAS is accepted to PRIME 2023 conference under the title: "A hardware-aware neural architecture search algorithm targeting low-end microcontrollers".

# How to use
* In search.py modify: 
  * "path_to_training_set" and "path_to_test_set" variables to use your own dataset (a small-dataset suggestion: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download).
  * "input_size" to set the network's input size
  * "params_upper_bound" and "MACC_upper_bound" to set the search's upper bounds
* Run search.py

In the folder "results", you will find a copy of the trained Keras model and the corresponding fully quantized Tflite model at uint8, ready to run on a microcontroller.

**Hint**: try multiple runs to find the best result.

# Models 
In the "Models" folder, you will find the models proposed in the paper "A hardware-aware neural architecture search algorithm targeting low-end microcontrollers", the search histories and the script utilized to build the Visual Wake Word dataset.

# Requirement
* Python 3.9.15
* Tensorflow 2.11.0
