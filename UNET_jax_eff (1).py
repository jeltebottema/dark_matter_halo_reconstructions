import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.signal import convolve as convolve3d

class PeriodicPadding3D(nn.Module):
    def __call__(self, x):
        # Optimized padding with minimal memory overhead
        return jnp.pad(x, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode='wrap')

class SpatialDropout3D(nn.Module):
    rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if deterministic or self.rate == 0:
            return x
        else:
            keep_prob = 1.0 - self.rate
            dropout_rng = self.make_rng('dropout')
            # Optimized mask generation
            random_mask = jax.random.bernoulli(dropout_rng, keep_prob, (x.shape[0], x.shape[-1]))
            scale = (1.0 / keep_prob)
            return x * random_mask[..., None, None, None, :]

class UpSampling3D(nn.Module):
    size: tuple

    def __call__(self, x):
        # Memory efficient upsampling
        return jax.image.resize(x, (x.shape[0], x.shape[1] * self.size[0], x.shape[2] * self.size[1], x.shape[3] * self.size[2], x.shape[4]), method='nearest')


        
def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3), strides=(1, 1, 1), training=True):
    conv = nn.Conv(features=n_filters, kernel_size=kernel, strides=strides, padding='VALID')(input_layer)
    norm = nn.BatchNorm(use_running_average=not training)(conv)
    return nn.leaky_relu(norm)

def create_localization_module(input_layer, n_filters, training=True):
    layer1 = PeriodicPadding3D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters, training=training)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1), training=training)
    return convolution2

def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2), training=True):
    up_sample = UpSampling3D(size=size)(input_layer)
    layer1 = PeriodicPadding3D()(up_sample)
    convolution = create_convolution_block(layer1, n_filters, training=training)
    return convolution

def create_context_module(input_layer, n_level_filters, dropout_rate, training=True):
    layer1 = PeriodicPadding3D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_level_filters, training=training)
    dropout = SpatialDropout3D(rate=dropout_rate)(convolution1, deterministic=not training)
    layer2 = PeriodicPadding3D()(dropout)
    convolution2 = create_convolution_block(layer2, n_level_filters, training=training)
    return convolution2


import jax
import jax.numpy as jnp
from flax import linen as nn

class DisplacementTensors(nn.Module):
    BoxSize: float

    @nn.compact
    def __call__(self, inputs, running_average=False):
        grid = inputs.shape[1]
        # Meshgrid creation
        kx, ky, kz = jnp.meshgrid(
            2 * jnp.pi * jnp.fft.fftfreq(grid, self.BoxSize / grid),
            2 * jnp.pi * jnp.fft.fftfreq(grid, self.BoxSize / grid),
            2 * jnp.pi * jnp.fft.rfftfreq(grid, self.BoxSize / grid),
            indexing="ij", sparse=True
        )

        # Convert to complex and compute knorm2
        knorm2 = jnp.maximum(kx**2 + ky**2 + kz**2, 1e-7) 

        # Perform FFT and scaling
        inputs_fft = jnp.fft.rfftn(inputs, axes=(1, 2, 3))
        scaled_fft = inputs_fft * 1j / knorm2[..., None]  

        # Computation of displacement tensors
        psix = scaled_fft * kx[..., None]
        psiy = scaled_fft * ky[..., None]
        psiz = scaled_fft * kz[..., None]
        psixy = psix * ky[..., None] * 1j
        psixz = psix * kz[..., None] * 1j
        psiyz = psiy * kz[..., None] * 1j

        # Concatenate tensors along the last dimension
        outputs_fft = jnp.concatenate([psix, psiy, psiz, psixy, psixz, psiyz], axis=-1)

        # Perform inverse FFT and adjust dimensions
        output = jnp.fft.irfftn(outputs_fft, s=inputs.shape[1:4], axes=(1, 2, 3))

        return output



        
class UNET3D_jax_e(nn.Module):
    image_size: int
    BoxSize: float
    n_base_filters: int = 16
    depth: int = 5
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, inputs, training=True):
        # x = inputs
        x = DisplacementTensors(self.BoxSize)(inputs)
        x = jnp.concatenate([inputs, x], axis=-1)

        level_output_layers = []
        level_filters = []
        current_grid = self.image_size

        for level_number in range(self.depth):
            n_level_filters = (2 ** level_number) * self.n_base_filters
            level_filters.append(n_level_filters)

            if level_number == 0:
                x = PeriodicPadding3D()(x)
                x = create_convolution_block(x, n_level_filters, training=training)
            else:
                x = PeriodicPadding3D()(x)
                x = create_convolution_block(x, n_level_filters, strides=(2, 2, 2), training=training)
                current_grid //= 2
            previous_block = x
            x = create_context_module(x, n_level_filters, dropout_rate = self.dropout_rate, training=training)
            x = previous_block + x
            level_output_layers.append(x)

        for level_number in range(self.depth - 2, -1, -1):
            current_grid *= 2
            x = create_up_sampling_module(x,  level_filters[level_number], training=training)
            x = jnp.concatenate([level_output_layers[level_number], x], axis=-1)
            x = create_localization_module(x,level_filters[level_number], training=training)

        x = nn.Conv(features=1, kernel_size=(1, 1, 1))(x)
        return x