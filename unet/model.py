from keras import layers, models

# Returns the encoder block with pooling, and without pooling
def encoder_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None)(append_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    # create a copy of the layer without pooling for skip connections
    skip_conn = x
    # Pool
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    return x, skip_conn

def bottleneck_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, num_conv:int=1, padding:str='same'):
    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None)(append_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    return x

def decoder_block(kernel_size:tuple, activation:str, layer_size:int, append_layer, skip_layer, num_conv:int=1, padding:str='same'):
    # Upsample
    x = layers.UpSampling2D(size=(2,2))(append_layer)

    x = layers.concatenate([skip_layer, x], axis=3)

    x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    # Add skip connection
    
    # Any additional convolutions
    for _ in range(num_conv - 1):
        x = layers.Conv2D(layer_size, kernel_size=kernel_size, padding=padding, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    return x

def create_unet(input_shape, num_classes, 
               initial_filters=64, max_depth=3, conv_per_block:int=1, 
               kernel_size:tuple=(3,3), activation:str='relu', padding:str='same'):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    skip_conns = []

    # Encoder
    for curr_depth in range(max_depth-1):
       layer, skip_layer = encoder_block(kernel_size=kernel_size, activation=activation,
                             layer_size=initial_filters * 2 ** (curr_depth), append_layer=x, num_conv=conv_per_block,
                             padding=padding)
       x = layer
       skip_conns.append(skip_layer)
    
    # Bottleneck
    x = bottleneck_block(kernel_size=kernel_size, activation=activation,
                             layer_size=initial_filters * 2 ** (max_depth-1), append_layer=x, num_conv=conv_per_block,
                             padding=padding)
    
    # Decoder
    for curr_depth in reversed(range(max_depth-1)):
        x = decoder_block(kernel_size=kernel_size, activation=activation,
                      layer_size=initial_filters * 2 ** (curr_depth), append_layer=x, skip_layer=skip_conns.pop(), num_conv=conv_per_block,
                      padding=padding)

    decoder_output = layers.Conv2D(num_classes, kernel_size=kernel_size, activation='softmax', padding=padding)(x)

    model = models.Model(inputs=inputs, outputs=decoder_output)
    return model