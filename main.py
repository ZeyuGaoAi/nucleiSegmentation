from model import *
from data import *
from segmentation_models import Unet

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect')
myGene = trainGenerator(2,'data/MoNuSeg/train','image_crop','label_crop',data_gen_args,image_color_mode='rgb',target_size = (512,512), save_to_dir=None)

#num_batch = 1
#for i,batch in enumerate(myGene):
#    if(i >= num_batch):
#        break

## original unet without pretrained
#model = unet(input_size=(512,512,3), lr = 0.0001)
## resnet34 or other unet with pretrained
model = Unet('resnet34', encoder_weights='imagenet')
model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

model_checkpoint = ModelCheckpoint('unet_MoNuSeg.hdf5', monitor='loss',verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, mode='auto',verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
model.fit_generator(myGene,steps_per_epoch=200,epochs=1000,callbacks=[model_checkpoint, reduce_lr, early_stopping])


testGene = testGenerator("data/MoNuSeg/test", as_gray = False)
results = model.predict_generator(testGene, 7, verbose=1)
saveResult("data/MoNuSeg/test",results)
