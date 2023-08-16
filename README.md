# NanoNAS
A small hardware-aware neural architecture search (HW NAS) algorithm targeting low-RAM microcontrollers. Given its low search cost, it can be executed on laptops without a GPU.

It provides small CNNs that can fit the constraints of low-RAM microcontrollers. 

# News
* **2023/08** NanoNAS will be presented in an [invited talk](https://www.tinyml.org/event/tinyml-talks-a-hardware-aware-neural-architecture-search-algorithm-targeting-ultra-low-power-microcontrollers) by the tinyML Foundation. Here is the [link](https://us02web.zoom.us/webinar/register/2216905294136/WN_fQRnR2SuQzuEAqhvO-AWeg#/registration) for the registration to the webinar.
* **2023/07** NanoNAS has been **updated**: now you can directly put the **RAM** and **Flash** available on your microcontroller as **search constraints**.
* **2023/04** NanoNAS is accepted to the PRIME 2023 conference under the title: "A hardware-aware neural architecture search algorithm targeting low-end microcontrollers".

# An overview of its performances

This section shows models obtained by running NanoNAS on several targets using the [Visual Wake Words dataset](https://arxiv.org/abs/1906.05721), a [standard TinyML benchmark](https://arxiv.org/abs/2003.04821). The following table shows the models' accuracy over the test set (the mini-val split), RAM and FLASH occupancy, and the corresponding hardware target. The search cost (search time + training time) is also reported in the table. It has been measured on a laptop featuring an 11th Gen Intel(R) Core(TM) i7-11370H CPU @ 3.30GHz equipped with 16 GB of RAM and 512 GB of SSD, without using a GPU.

| Target              | Model Name              | Accuracy | RAM occupancy | FLASH occupancy | Resolution   | Search Cost | GPU   |
| :---                |    :---                 |  :---:   |     :---:     |     :---:       |    :---:     |    :---:    | :---: |
| NUCLEO-L010RB       | [vww_nucleo_l010rb](https://github.com/AndreaMattiaGaravagno/NanoNAS/blob/main/Models/performance_overview/vww_nucleo_l010rb.tflite)       | 72.3%    |    20 kiB     |     10.66 kiB   |    50x50 rgb |    1:50h    | no    |
| Arduino Nano 33 IoT | [vww_arduino_nano_33_iot](https://github.com/AndreaMattiaGaravagno/NanoNAS/blob/main/Models/performance_overview/vww_arduino_nano_33_iot.tflite) | 74.6%    |    26 kiB     |     19.73 kiB   |    50x50 rgb |    2:01h    | no    |
| NUCLEO-L412KB       | [vww_nucleo_l412kb](https://github.com/AndreaMattiaGaravagno/NanoNAS/blob/main/Models/performance_overview/vww_nucleo_l412kb.tflite)       | 77.2%    |    31 kiB     |     28.48 kiB   |    50x50 rgb |    3:53h    | no    |

[Here](https://github.com/AndreaMattiaGaravagno/NanoNAS/blob/main/Models/PRIME23/build_visual_wake_words_dataset.py) is a script for building the Visual Wake Words dataset.

Considering the smallest model offered by two state-of-the-art HW-NAS targeting microcontrollers, [Micronets](https://arxiv.org/pdf/2010.11267.pdf) and [MCUNET](https://arxiv.org/abs/2007.10319), and running NanoNAS using the constraints of the larger model of the twos, we can see that NanoNAS delivers significantly smaller networks both in terms of RAM and FLASH occupancy while achieving competitive accuracy over the Visual Wake Words dataset, as shown in the table below.

| Project   | Model Name          | Accuracy | RAM occupancy | FLASH occupancy | Resolution      | Search Cost | GPU   |
| :---      |  :---               |  :---:   |     :---:     |      :---:      |   :---:         |    :---:    | :---: |
| Micronets | [MicroNet VWW-2 INT8](https://github.com/ARM-software/ML-zoo/tree/master/models/visual_wake_words/micronet_vww2/tflite_int8) | 76.8%    | 70.50 kiB     | 273.81 kiB      | 50x50 grayscale |     n/a     | yes   |
| NanoNAS   | [vww-PRIME23](https://github.com/AndreaMattiaGaravagno/NanoNAS/blob/main/Models/PRIME23/visual_wake_words.tflite)          | 77%      | 28.50 kiB     | 23.65 kiB       | 50x50 rgb       |    3:37h    | no    |
| MCUNET    | [mcunet-vww0](https://github.com/mit-han-lab/mcunet)         | 87.4%    | 168.5 kiB     | 530.52 kiB      | 64x64 rgb       |    300h     | yes   |

For further details of the comparison refer to "A hardware-aware neural architecture search algorithm targeting low-end microcontrollers" by A.M. Garavagno et al. published in the proceedings of the 18th Conference on Ph. D Research in Microelectronics and Electronics (PRIME), 2023.

# How to use

* In search.py modify: 
  * "path_to_training_set" and "path_to_test_set" variables to use your own dataset ([a small-dataset suggestion](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download))
  * "input_size" to set the network's input size (50x50 is a good starting point for microcontrollers)
  * "RAM_upper_bound" and "Flash_upper_bound" according to your microcontroller
  * "MACC_upper_bound" according to the maximum desired number of MACC (a good starting point is given by multiplying the [CoreMark](https://www.eembc.org/coremark/) score of your microcontroller by 10000)
* Run search.py

In the folder "results", you will find a copy of the trained Keras model and the corresponding fully quantized Tflite model at uint8, ready to run on a microcontroller.

**Hint**: try multiple runs to find the best result.

# Requirement
* Python 3.9.15
* Tensorflow 2.11.0

# Citation
If you find the project helpful, please consider citing our paper:

    @inproceedings{garavagno2023hardware,
        title={A hardware-aware neural architecture search algorithm targeting low-end microcontrollers},
        author={Garavagno, Andrea Mattia and Ragusa, Edoardo and Frisoli, Antonio and Gastaldo, Paolo},
        booktitle={2023 18th Conference on Ph. D Research in Microelectronics and Electronics (PRIME)},
        pages={281--284},
        year={2023},
        organization={IEEE}
    }
