from keras import layers, models

# Create model
def encoder_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(append_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv-1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    return x

def bottleneck(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(append_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    return x

def decoder_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.UpSampling2D(size=(2,2))(append_layer)
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    return x

def create_autoencoder(input_shape, num_classes, 
               encoder_layer_sizes:list, bottleneck_size:int, decoder_layer_sizes:list,
               conv_per_block:int=1, kernel_size:tuple=(3,3), activation:str='relu', padding:str='same'):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for size in encoder_layer_sizes:
        x = encoder_block(kernel_size=kernel_size, activation=activation, layer_size=size, append_layer=x, num_conv=conv_per_block, padding=padding)

    x = bottleneck(kernel_size=kernel_size, activation=activation, layer_size=bottleneck_size, append_layer=x, num_conv=conv_per_block, padding=padding)

    for size in decoder_layer_sizes:
        x = decoder_block(kernel_size=kernel_size, activation=activation, layer_size=size, append_layer=x, num_conv=conv_per_block, padding=padding)

    decoder_output = layers.Conv2D(num_classes, kernel_size=kernel_size, activation='softmax', padding=padding)(x)

    model = models.Model(inputs=inputs, outputs=decoder_output)
    return model