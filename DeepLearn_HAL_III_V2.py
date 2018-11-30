import datetime
import json
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
import time
import os
import random


area = 33


# Build the convolutional NN (Based off previous Starcraft NN)
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=(area,area,3), activation='relu'))   
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(126, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))

learning_rate = 0.0001  # may need to adjust
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/HAL_III-{learning_rate}-{int(time.time())}')

# Iterate through data
train_directory = "training_data"
hm_epochs = 20
def check_data():
    choices = {"move_north": move_north,
                "move_south": move_south,
                "move_east": move_east,
                "move_west": move_west,
                "stay_still": stay_still}
    total_data = 0
    lengths = []
    for choice in choices:
        print(f"The length of {choice} is: {len(choices[choice])}")
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))
    
    print("Total data length now is: ",total_data)
    return lengths


for i in range(hm_epochs):
    print(f"Currently on epoch {i}")
    current = 0
    increment = 3200
    
    not_maximum = True
    all_files = os.listdir(train_directory)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        print(f"Currently doing {current}:{current+increment}")
        #  directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
        move_north = [] # 0 
        move_south = [] # 1
        move_east = [] # 2
        move_west = [] # 3
        stay_still = [] # 4

        for file in all_files[current:current+increment]:
            full_path = os.path.join(train_directory,file)
            data = np.load(full_path)
            data = list(data)
            for d in data:
                choice = d[1]
                if choice == 0:
                    converted_choice = [1,0,0,0,0]
                    move_north.append([d[0],converted_choice])
                elif choice == 1:
                    converted_choice = [0,1,0,0,0]
                    move_south.append([d[0],converted_choice])
                elif choice == 2:
                    converted_choice = [0,0,1,0,0]
                    move_east.append([d[0],converted_choice])
                elif choice == 3:
                    converted_choice = [0,0,0,1,0]
                    move_west.append([d[0],converted_choice])
                elif choice == 4:
                    converted_choice = [0,0,0,0,1]
                    stay_still.append([d[0],converted_choice])
            
        lengths = check_data()
        lowest_data = min(lengths)
        random.shuffle(move_north)
        random.shuffle(move_south)
        random.shuffle(move_east)
        random.shuffle(move_west)
        random.shuffle(stay_still)

        move_north = move_north[:lowest_data]
        move_south = move_south[:lowest_data]
        move_east = move_east[:lowest_data]
        move_west = move_west[:lowest_data]
        stay_still = stay_still[:lowest_data]

        check_data()
        training_data = move_north + move_south + move_east + move_west + stay_still
        random.shuffle(training_data)
        test_size = 100
        batch_size = 128

        
        x_train = np.array([i[0] for i in training_data[:-test_size]]).reshape(-1, 33, 33, 3)
        y_train = np.array([i[1] for i in training_data[:-test_size]])

        x_test = np.array([i[0] for i in training_data[-test_size:]]).reshape(-1, 33, 33, 3)
        y_test = np.array([i[1] for i in training_data[-test_size:]])

        model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=True, verbose=1, callbacks=[tensorboard])

        model.save(f"BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1")
        current += increment
        if current >= maximum:
            not_maximum = False

