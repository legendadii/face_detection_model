
import os
images="C:\\Users\\DELL\\Downloads\\archive (10)\\Face Mask Dataset"
folder=os.listdir("C:\\Users\\DELL\\Downloads\\archive (10)\\Face Mask Dataset")
folders=os.listdir("C:\\Users\\DELL\\Downloads\\archive (10)\\Face Mask Dataset\\train")
print(folder)

image_data=[]
labels=[]
j=0
label_dict={}
la = {}
la1 = {}


for i in folders:
    print(i)
    label_dict[i]=j
    j=j+1

print(label_dict)

from keras.preprocessing import image

for ix in folder:
    print(ix)
    path=images+ "\\\\" +ix
    print(path) 
    folder2=os.listdir(path)
    for iy in folder2:
        # print(iy)
        path1 =path + "\\\\" +iy
        print(path1) 
        for i in os.listdir(path1):
            img=image.load_img(os.path.join(path1,i),target_size=((224,224)))
            img_array=image.img_to_array(img)
            # img_array=img_array.astype("float32")/255
            image_data.append(img_array)
            labels.append(label_dict[iy])

    import numpy as np
    import random 
    combined=list(zip(image_data,labels))
    random.shuffle(combined)

    image_data[:],labels[:]=zip(*combined)
    la[ix] =np.array(image_data)
    la1[ix]=np.array(labels)
    print(la[ix].shape)
    print(la1[ix].shape)
    image_data=[]
    labels=[]


x_train=la["Train"]
y_train=la1["Train"]

x_test=la["Test"]
y_test=la1["Test"]

x_val=la["Validation"]
y_val=la1["Validation"]


import os
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
from sklearn import model_selection
from keras.models import Sequential, Model
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.optimizers import *
from keras.layers import *
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
# from keras import tunner
# import keras.tunner as kt
from keras.models import Model
from keras.layers import Dense
import kerastuner as kt


base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Initializing the VGG19 model

# desired_layer_name = 'conv_dw_13_bn'
# desired_layer_output = base_model.get_layer(desired_layer_name).output
# model1 = Model(inputs=base_model.input, outputs=desired_layer_output)
print("************************************************************************************")
def build_model(hp):
    model = Sequential()

    model.add(base_model)  
    model.add(Flatten())

    # base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # desired_layer_name = 'conv_dw_13_bn'
    # desired_layer_output = base_model.get_layer(desired_layer_name).output
    # model1 = Model(inputs=base_model.input, outputs=desired_layer_output)
    # model.add(model1)
    # model.add(Flatten())

 
    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=5, default=2)
    
    for i in range(num_dense_layers):
       
        units = hp.Int('units_' + str(i), min_value=32, max_value=512, step=32)
        
        
        activation = hp.Choice('activation_' + str(i), values=['relu', 'tanh', 'sigmoid'])
        
        model.add(Dense(units, activation=activation))
        
      
        dropout_rate = hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate))

   
    model.add(Dense(1, activation='sigmoid'))

   
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    return model


print("**************************************************")


tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='mydir2',
    project_name='final'
)


tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


best_hyperparameters = tuner.get_best_hyperparameters()[0]
print(best_hyperparameters.values)


best_model = tuner.hypermodel.build(best_hyperparameters)




from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode="auto")
early_stopping = EarlyStopping(monitor="val_accuracy", patience=4, min_delta=0.2, mode="auto")

# Train the final model
best_model.fit(x_train,y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=[checkpoint, early_stopping])

print("+++++++++++++++++++++++++++++++++++++++++++++")
print(best_model.evaluate(x_test,y_test)[1])
print(best_model.evaluate(x_test,y_test)[0])


h= best_model.history

dict_keys=[['loss','accuracy','val-loss','val-accuracy']]




import matplotlib.pyplot as plt
plt.plot[h['accuracy']]
plt.plot[h['val-accuracy']]
plt.show()

plt.plot[h['loss']]
plt.plot[h['val-loss']]
plt.show()


plt.plot[h['accuracy']]
plt.plot[h['loss']]
plt.show()

plt.plot[h['val-loss']]
plt.plot[h['val_accuracy']]
plt.show()





# Compiling the model
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Change loss to binary_crossentropy

# # Model Training
# model_history = model.fit(
#     x_train, y_train,  # Use y_train[:, 1] to extract the second column for binary classification
#     epochs=20,
#     shuffle=True,
#     batch_size=256,
#     validation_data=(x_val, y_val)  # Use y_val[:, 1] for validation data
# )

import pickle 
with open('model21_pickle','wb') as f:
    pickle.dump(best_model,f)

with open('model21_pickle','rb') as f:
    best_model=pickle.load(f)




model_loss, model_acc = best_model.evaluate(x_test,y_test)
print("Model has a loss of %.2f and accuracy %.2f%%" % (model_loss, model_acc*100))





import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("../input/face-mask-detection/images/maksssksksss352.png")

# Keep a copy of coloured image
orig_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image
plt.figure(figsize=(12, 12))
plt.imshow(orig_img)

# Convert image to grayscale
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)


# loading haarcascade_frontalface_default.xml
face_detection_model = cv2.CascadeClassifier("../input/haar-cascades-for-face-detection/haarcascade_frontalface_default.xml")

# detect faces in the given image
return_faces = face_detection_model.detectMultiScale(
    img, scaleFactor=1.08, minNeighbors=4
)  # returns a list of (x,y,w,h) tuples

# plotting the returned values
for (x, y, w, h) in return_faces:
    cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

plt.figure(figsize=(12, 12))
plt.imshow(orig_img)  # display the image


mask_det_label = {0: "Mask", 1: "No Mask"}
mask_det_label_colour = {0: (0, 255, 0), 1: (255, 0, 0)}
pad_y = 1  # padding for result text

main_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

# For detected faces in the image
for i in range(len(return_faces)):
    (x, y, w, h) = return_faces[i]
    cropped_face = main_img[y : y + h, x : x + w]
    cropped_face = cv2.resize(cropped_face, (128, 128))
    cropped_face = np.reshape(cropped_face, [1, 128, 128, 3]) / 255.0
    mask_result = best_model.predict(cropped_face)  # make model prediction
    print_label = mask_det_label[mask_result.argmax()] # get mask/no mask based on prediction
    label_colour = mask_det_label_colour[mask_result.argmax()] # green for mask, red for no mask

    # Print result
    (t_w, t_h), _ = cv2.getTextSize(
        print_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    )  # getting the text size
    
    cv2.rectangle(
        main_img,
        (x, y + pad_y),
        (x + t_w, y - t_h - pad_y - 6),
        label_colour,
        -1,
    )  # draw rectangle

    cv2.putText(
        main_img,
        print_label,
        (x, y - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        (255, 255, 255), # white
        1,
    )  # print text

    cv2.rectangle(
        main_img,
        (x, y),
        (x + w, y + h),
        label_colour,
        1,
    )  # draw bounding box on face

plt.figure(figsize=(10, 10))
plt.imshow(main_img)  # display image


