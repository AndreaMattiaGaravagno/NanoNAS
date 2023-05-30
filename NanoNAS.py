from multiprocessing import Process, Queue
import tensorflow as tf
import numpy as np
import datetime
import os

class NanoNAS :
    architecture_name = 'resulting_architecture'
    def __init__(self, max_MACC, max_params, path_to_training_set, val_split, path_to_test_set, cache=False, input_shape=(50,50,3), save_path='./') :
        self.path_to_training_set = path_to_training_set
        self.num_classes = len(next(os.walk(path_to_training_set))[1])
        self.val_split = val_split
        self.input_shape = input_shape
        self.max_MACC = max_MACC
        self.max_params = max_params
        self.cache = cache
        self.save_path = save_path
        self.path_to_test_set = path_to_test_set

    def get_params(self, model) :
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams

        return totalParams

    # k number of kernels of the first convolutional layer
    # c number of cells added upon the first convolutional layer
    # pre-processing pipeline not included in MACC computation
    def Model(self, k, c) :
        kernel_size = (3,3)
        pool_size = (2,2)
        pool_strides = (2,2)

        number_of_cells_limited = False
        number_of_mac = 0

        inputs = tf.keras.Input(shape=self.input_shape)

        #preprocessing pipeline
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        #convolutional base
        n = k
        multiplier = 2

        #first convolutional layer
        c_in = self.input_shape[2]
        x = tf.keras.layers.Conv2D(n, kernel_size, activation='relu', padding='same')(x)
        number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

        #adding cells
        for i in range(1, c + 1) :
            if x.shape[1] <= 1 or x.shape[2] <= 1 :
                number_of_cells_limited = True
                break;
            n = np.ceil(n * multiplier)
            multiplier = multiplier - 2**-i
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            c_in = x.shape[3]
            x = tf.keras.layers.Conv2D(n, kernel_size, activation='relu', padding='same')(x)
            number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

        #classifier
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        number_of_mac = number_of_mac + (x.shape[1] * outputs.shape[1])

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model, number_of_mac, self.get_params(model), number_of_cells_limited

    def load_training_set(self, batch_size=1):
        if 3 == self.input_shape[2] :
            color_mode = 'rgb'
        elif 1 == self.input_shape[2] :
            color_mode = 'grayscale'

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory= self.path_to_training_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True,
            seed=11,
            validation_split=self.val_split,
            subset='training'
        )

        validation_ds = tf.keras.utils.image_dataset_from_directory(
            directory= self.path_to_training_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True,
            seed=11,
            validation_split=self.val_split,
            subset='validation'
        )

        if self.cache :
            train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        else :
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, validation_ds

    def compile_model(self, model, learning_rate):
         opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

         model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    def evaluate_model_process(self, q, k, c) :
        epochs = 3
        search_learning_rate = 0.001
        search_batch_size = 16

        train_ds, validation_ds = self.load_training_set(search_batch_size)

        model, number_of_mac, number_of_params, number_of_cells_limited = self.Model(k, c)
        if number_of_mac <= self.max_MACC and number_of_params <= self.max_params and not number_of_cells_limited :
            self.compile_model(model, search_learning_rate)
            hist = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, validation_freq=1)
            if (self.save_search_history) :
                model_name = 'k' + str(k) + '_c' + str(c)
                new_dir = self.save_path + '/search_history/search_learning_rate_' + str(search_learning_rate) + '/search_batch_size_' + str(search_batch_size) + '/' + model_name + '/'
                os.makedirs(new_dir)
                np.save(new_dir + model_name + '_hist.npy', hist.history)
            q.put({'k': k, 'c': c, 'score': np.around(np.amax(hist.history['val_accuracy']), decimals=3)})
        else :
            q.put({'k': k, 'c': c, 'score': -3})

    def search(self, save_search_history=False) :
        self.save_search_history = save_search_history

        start = datetime.datetime.now()

        best_architecture = {'k': -1, 'c': -1, 'score': -2}
        new_architecture = {'k': -1, 'c': -1, 'score': -1}

        k = 1
        while(new_architecture['score'] > best_architecture['score']) :
            best_architecture = new_architecture
            c = 0
            previous_architecture = {'k': -1, 'c': -1, 'score': -2}
            current_architecture = {'k': -1, 'c': -1, 'score': -1}
            while(current_architecture['score'] > previous_architecture['score']) :
                previous_architecture = current_architecture
                c = c + 1
                q = Queue()
                p = Process(target=self.evaluate_model_process, args=(q, k, c,))
                p.start()
                current_architecture = q.get()
                p.join()
                print(f"\n{current_architecture}\n")
            new_architecture = previous_architecture
            k = k + 1

        end = datetime.datetime.now()
        print(f"\nResulting architecture: {best_architecture}\n")
        print(f"Elapsed time (search): {end-start}\n")

        self.resulting_architecture = best_architecture

    def train_process(self, q, training_epochs, training_learning_rate, training_batch_size) :
        train_ds, validation_ds = self.load_training_set(training_batch_size)
        model = self.Model(self.resulting_architecture['k'], self.resulting_architecture['c'])[0]
        self.compile_model(model, training_learning_rate)
        path_to_keras_model = self.save_path + '/' + self.architecture_name + '.h5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath= path_to_keras_model,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        hist = model.fit(train_ds, epochs=training_epochs, validation_data=validation_ds, validation_freq=1, callbacks=[model_checkpoint_callback])

        print('\nmax val acc: ' + str(round(np.amax(hist.history['val_accuracy']), 3)))

        q.put(path_to_keras_model)

    def train(self, training_epochs, training_learning_rate, training_batch_size) :
        start = datetime.datetime.now()
        q = Queue()
        p = Process(target=self.train_process, args=((q, training_epochs, training_learning_rate, training_batch_size,)))
        p.start()
        self.path_to_keras_model = q.get()
        p.join()
        end = datetime.datetime.now()
        print(f"\nKeras model saved in: {os.path.abspath(self.path_to_keras_model)}\n")
        print(f"Elapsed time (training): {end-start}\n")

    def apply_uint8_post_training_quantization_process(self, q) :

        def representative_dataset():
            for data in train_ds.rebatch(1).take(150) :
                yield [tf.dtypes.cast(data[0], tf.float32)]

        train_ds, validation_ds = self.load_training_set()

        model = tf.keras.models.load_model(self.path_to_keras_model)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        tflite_quant_model = converter.convert()

        path_to_tflite_model = self.save_path + '/' + self.architecture_name + '.tflite'

        with open(path_to_tflite_model, 'wb') as f:
            f.write(tflite_quant_model)

        q.put(path_to_tflite_model)

    def apply_uint8_post_training_quantization(self) :
        q = Queue()
        p = Process(target=self.apply_uint8_post_training_quantization_process, args=(q,))
        p.start()
        self.path_to_tflite_model = q.get()
        p.join()
        print(f"\nTflite model saved in: {os.path.abspath(self.path_to_tflite_model)}\n")

    def load_test_set(self, batch_size=1):
        if 3 == self.input_shape[2] :
            color_mode = 'rgb'
        elif 1 == self.input_shape[2] :
            color_mode = 'grayscale'

        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory= self.path_to_test_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True,
            seed=11,
            validation_split=self.val_split,
            subset='training'
        )

        if self.cache :
            test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        else :
            test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return test_ds

    def test_keras_model_process(self) :
        test_ds = self.load_test_set()
        model = tf.keras.models.load_model(self.path_to_keras_model)

        #evaluate keras model
        print(f"\nKeras model test accuracy: {model.evaluate(test_ds)[1]}\n")

    def test_keras_model(self) :
        p = Process(target=self.test_keras_model_process, args=())
        p.start()
        p.join()

    def test_tflite_model_process(self) :
        test_ds = self.load_test_set()
        interpreter = tf.lite.Interpreter(self.path_to_tflite_model)
        interpreter.allocate_tensors()

        output = interpreter.get_output_details()[0]  # Model has single output.
        input = interpreter.get_input_details()[0]  # Model has single input.

        correct = 0
        wrong = 0

        for image, label in test_ds.rebatch(1) :
            # Check if the input type is quantized, then rescale input data to uint8
            if input['dtype'] == tf.uint8:
                input_scale, input_zero_point = input["quantization"]
                image = image / input_scale + input_zero_point
            input_data = tf.dtypes.cast(image, tf.uint8)
            interpreter.set_tensor(input['index'], input_data)
            interpreter.invoke()
            if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
                correct = correct + 1
            else :
                wrong = wrong + 1
        print(f"\nTflite model test accuracy: {correct/(correct+wrong)}")

    def test_tflite_model(self) :
        p = Process(target=self.test_tflite_model_process, args=())
        p.start()
        p.join()
