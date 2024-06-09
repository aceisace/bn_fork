# BlueNoise.py - An implementation of the void and cluster method for generation of
#                blue noise dither arrays and related utilities.
#
# Written in 2016 by Christoph Peters, Christoph(at)MomentsInGraphics.de
#
# To the extent possible under law, the author(s) have dedicated all copyright and
# related and neighboring rights to this software to the public domain worldwide.
# This software is distributed without any warranty.
#
# You should have received a copy of the CC0 Public Domain Dedication along with
# this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

# edited by aceisace in 2024 for support of Python 3.9 and above and updating deprecated functions

import os
import struct
import threading

import numpy as np
import png
from matplotlib import pyplot
from scipy.ndimage import fourier_gaussian


def get_bayer_pattern(log_2_width):
    """Creates a two-dimensional Bayer pattern with a width and height of
       2**Log2Width."""
    x, y = np.meshgrid(range(2 ** log_2_width), range(2 ** log_2_width))
    result = np.zeros_like(x)
    for i in range(log_2_width):
        stripes_y = np.where(np.bitwise_and(y, 2 ** (log_2_width - 1 - i)) != 0, 1, 0)
        stripes_x = np.where(np.bitwise_and(x, 2 ** (log_2_width - 1 - i)) != 0, 1, 0)
        checker = np.bitwise_xor(stripes_x, stripes_y)
        result += np.bitwise_or(stripes_y * 2 ** (2 * i), checker * 2 ** (2 * i + 1))
    return result


def find_largest_void(binary_pattern, standard_deviation):
    """This function returns the indices of the largest void in the given binary
       pattern as defined by Ulichney.
      \param BinaryPattern A boolean array (should be two-dimensional although the
             implementation works in arbitrary dimensions).
      \param StandardDeviation The standard deviation used for the Gaussian filter
             in pixels. This can be a single float for an isotropic Gaussian or a
             tuple with one float per dimension for an anisotropic Gaussian.
      \return A flat index i such that BinaryPattern.flat[i] corresponds to the
              largest void. By definition this is a majority pixel.
      \sa GetVoidAndClusterBlueNoise"""
    # The minority value is always True for convenience
    if np.count_nonzero(binary_pattern) * 2 >= np.size(binary_pattern):
        binary_pattern = np.logical_not(binary_pattern)
    # Apply the Gaussian. We do not want to cut off the Gaussian at all because even
    # the tiniest difference can change the ranking. Therefore we apply the Gaussian
    # through a fast Fourier transform by means of the convolution theorem.
    FilteredArray = np.fft.ifftn(
        fourier_gaussian(np.fft.fftn(np.where(binary_pattern, 1.0, 0.0)), standard_deviation)).real
    # Find the largest void
    return np.argmin(np.where(binary_pattern, 2.0, FilteredArray))


def find_tightest_cluster(binary_pattern, standard_deviation):
    """Like FindLargestVoid() but finds the tightest cluster which is a minority
       pixel by definition.
      \sa GetVoidAndClusterBlueNoise"""
    if np.count_nonzero(binary_pattern) * 2 >= np.size(binary_pattern):
        binary_pattern = np.logical_not(binary_pattern)
    FilteredArray = np.fft.ifftn(
        fourier_gaussian(np.fft.fftn(np.where(binary_pattern, 1.0, 0.0)), standard_deviation)).real
    return np.argmax(np.where(binary_pattern, FilteredArray, -1.0))


def get_void_and_cluster_blue_noise(OutputShape, standard_deviation=1.5, initial_seed_fraction=0.1):
    """Generates a blue noise dither array of the given shape using the method
       proposed by Ulichney [1993] in "The void-and-cluster method for dither array
       generation" published in Proc. SPIE 1913.
      \param OutputShape The shape of the output array. This function works in
             arbitrary dimension, i.e. OutputShape can have arbitrary length. Though
             it is only tested for the 2D case where you should pass a tuple
             (Height,Width).
      \param StandardDeviation The standard deviation in pixels used for the
             Gaussian filter defining largest voids and tightest clusters. Larger
             values lead to more low-frequency content but better isotropy. Small
             values lead to more ordered patterns with less low-frequency content.
             Ulichney proposes to use a value of 1.5. If you want an anisotropic
             Gaussian, you can pass a tuple of length len(OutputShape) with one
             standard deviation per dimension.
      \param InitialSeedFraction The only non-deterministic step in the algorithm
             marks a small number of pixels in the grid randomly. This parameter
             defines the fraction of such points. It has to be positive but less
             than 0.5. Very small values lead to ordered patterns, beyond that there
             is little change.
      \return An integer array of shape OutputShape containing each integer from 0
              to np.prod(OutputShape)-1 exactly once."""
    n_rank = np.prod(OutputShape)
    # Generate the initial binary pattern with a prescribed number of ones
    n_initial_one = max(1, min(int((n_rank - 1) / 2), int(n_rank * initial_seed_fraction)))
    # Start from white noise (this is the only randomized step)
    initial_binary_pattern = np.zeros(OutputShape, dtype=bool)
    initial_binary_pattern.flat = np.random.permutation(np.arange(n_rank)) < n_initial_one
    # Swap ones from tightest clusters to largest voids iteratively until convergence
    while True:
        i_tightest_cluster = find_tightest_cluster(initial_binary_pattern, standard_deviation)
        initial_binary_pattern.flat[i_tightest_cluster] = False
        i_largest_void = find_largest_void(initial_binary_pattern, standard_deviation)
        if i_largest_void == i_tightest_cluster:
            initial_binary_pattern.flat[i_tightest_cluster] = True
            # Nothing has changed, so we have converged
            break
        else:
            initial_binary_pattern.flat[i_largest_void] = True
    # Rank all pixels
    dither_array = np.zeros(OutputShape, dtype=int)
    # Phase 1: Rank minority pixels in the initial binary pattern
    binary_pattern = np.copy(initial_binary_pattern)
    for rank in range(n_initial_one - 1, -1, -1):
        i_tightest_cluster = find_tightest_cluster(binary_pattern, standard_deviation)
        binary_pattern.flat[i_tightest_cluster] = False
        dither_array.flat[i_tightest_cluster] = rank
    # Phase 2: Rank the remainder of the first half of all pixels
    binary_pattern = initial_binary_pattern
    for rank in range(n_initial_one, int((n_rank + 1) / 2)):
        i_largest_void = find_largest_void(binary_pattern, standard_deviation)
        binary_pattern.flat[i_largest_void] = True
        dither_array.flat[i_largest_void] = rank
    # Phase 3: Rank the last half of pixels
    for rank in range(int((n_rank + 1) / 2), n_rank):
        i_tightest_cluster = find_tightest_cluster(binary_pattern, standard_deviation)
        binary_pattern.flat[i_tightest_cluster] = True
        dither_array.flat[i_tightest_cluster] = rank
    return dither_array


def analyze_noise_texture(texture, single_figure=True, simple_labels=False):
    """Given a 2D array of real noise values this function creates one or more
       figures with plots that allow you to analyze it, especially with respect to
       blue noise characteristics. The analysis includes the absolute value of the
       Fourier transform, the power distribution in radial frequency bands and an
       analysis of directional isotropy.
      \param A two-dimensional array.
      \param SingleFigure If this is True, all plots are shown in a single figure,
             which is useful for on-screen display. Otherwise one figure per plot
             is created.
      \param SimpleLabels Pass True to get axis labels that fit into the context of
             the blog post without further explanation.
      \return A list of all created figures.
      \note For the plots to show you have to invoke pyplot.show()."""
    figure_list = list()
    if single_figure:
        Figure = pyplot.figure()
        figure_list.append(Figure)

    def prepare_axes(i_axes, **kwargs):
        if single_figure:
            return Figure.add_subplot(2, 2, i_axes, **kwargs)
        else:
            new_figure = pyplot.figure()
            figure_list.append(new_figure)
            return new_figure.add_subplot(1, 1, 1, **kwargs)

    # Plot the dither array itself
    prepare_axes(1, title="Blue noise dither array")
    pyplot.imshow(texture.real, cmap="gray", interpolation="nearest")
    # Plot the Fourier transform with frequency zero shifted to the center
    prepare_axes(2, title="Fourier transform (absolute value)", xlabel="$\\omega_x$", ylabel="$\\omega_y$")
    dft = np.fft.fftshift(np.fft.fft2(texture)) / float(np.size(texture))
    height, width = texture.shape
    shift_y, shift_x = (int(height / 2), int(width / 2))
    pyplot.imshow(np.abs(dft), cmap="viridis", interpolation="nearest", vmin=0.0, vmax=np.percentile(np.abs(dft), 99),
                  extent=(-shift_x - 0.5, width - shift_x - 0.5, -shift_y + 0.5, height - shift_y + 0.5))
    pyplot.colorbar()
    # Plot the distribution of power over radial frequency bands
    prepare_axes(3, title="Radial power distribution",
                 xlabel="Distance from center / pixels" if simple_labels else "$\\sqrt{\\omega_x^2+\\omega_y^2}$")
    X, Y = np.meshgrid(range(dft.shape[1]), range(dft.shape[0]))
    X -= int(dft.shape[1] / 2)
    Y -= int(dft.shape[0] / 2)
    radial_frequency = np.asarray(np.round(np.sqrt(X ** 2 + Y ** 2)), dtype=int)
    radial_power = np.zeros((np.max(radial_frequency) - 1,))
    dft[int(dft.shape[0] / 2), int(dft.shape[1] / 2)] = 0.0
    for i in range(radial_power.shape[0]):
        radial_power[i] = np.sum(np.where(radial_frequency == i, np.abs(dft), 0.0)) / np.count_nonzero(
            radial_frequency == i)
    pyplot.plot(np.arange(np.max(radial_frequency) - 1) + 0.5, radial_power)
    # Plot the distribution of power over angular frequency ranges
    prepare_axes(4, title="Anisotropy (angular power distribution)", aspect="equal",
                 xlabel="Frequency x" if simple_labels else "$\\omega_x$",
                 ylabel="Frequency y" if simple_labels else "$\\omega_y$")
    circular_mask = np.logical_and(0 < radial_frequency, radial_frequency < int(min(dft.shape[0], dft.shape[1]) / 2))
    normalized_x = np.asarray(X, dtype=float) / np.maximum(1.0, np.sqrt(X ** 2 + Y ** 2))
    normalized_y = np.asarray(Y, dtype=float) / np.maximum(1.0, np.sqrt(X ** 2 + Y ** 2))
    binning_angle = np.linspace(0.0, 2.0 * np.pi, 33)
    angular_power = np.zeros_like(binning_angle)
    for i, angle in enumerate(binning_angle):
        dot_product = normalized_x * np.cos(angle) + normalized_y * np.sin(angle)
        FullMask = np.logical_and(circular_mask, dot_product >= np.cos(np.pi / 32.0))
        angular_power[i] = np.sum(np.where(FullMask, np.abs(dft), 0.0)) / np.count_nonzero(FullMask)
    MeanAngularPower = np.mean(angular_power[1:])
    DenseAngle = np.linspace(0.0, 2.0 * np.pi, 256)
    pyplot.plot(np.cos(DenseAngle) * MeanAngularPower, np.sin(DenseAngle) * MeanAngularPower, color=(0.7, 0.7, 0.7))
    pyplot.plot(np.cos(binning_angle) * angular_power, np.sin(binning_angle) * angular_power)
    return figure_list


def plot_binary_patterns(texture, n_pattern_row, n_pattern_column):
    """This function creates a figure with a grid of thresholded versions of the
       given 2D noise texture. It assumes that each value from 0 to
       np.size(Texture)-1 is contained exactly once.
      \return The created figure.
      \note For the plots to show you have to invoke pyplot.show()."""
    figure = pyplot.figure()
    n_pattern = n_pattern_row * n_pattern_column + 1
    for i in range(1, n_pattern):
        figure.add_subplot(n_pattern_row, n_pattern_column, i, xticks=[], yticks=[])
        pyplot.imshow(np.where(texture * n_pattern < i * np.size(texture), 1.0, 0.0), cmap="gray",
                      interpolation="nearest")
    return figure


def store_noise_texture_ldr(texture, output_png_file_path, n_rank=-1):
    """This function stores the given texture to a standard low-dynamic range png
       file with four channels and 8 bits per channel.
      \param Texture An array of shape (Height,Width) or (Height,Width,nChannel).
             The former is handled like (Height,Width,1). If nChannel>4 the
             superfluous channels are ignored. If nChannel<4 the data is expanded.
             The alpha channel is set to 255, green and blue are filled with black
             or with duplicates of red if nChannel==1. It is assumed that each
             channel contains every integer value from 0 to nRank-1 exactly once.
             The range of values is remapped linearly to span the range from 0 to
             255.
      \param OutputPNGFilePath The path to the output png file including the file
             format extension.
      \param nRank Defaults to Width*Height if you pass a non-positive value."""
    # Scale the array to an LDR version
    if n_rank <= 0:
        n_rank = texture.shape[0] * texture.shape[1]
    texture = np.asarray((texture * 256) // n_rank, dtype=np.uint8)
    # Get a three-dimensional array
    if len(texture.shape) < 3:
        texture = texture[:, :, np.newaxis]
    # Generate channels as needed
    if texture.shape[2] == 1:
        texture = np.dstack([texture] * 3 + [255 * np.ones_like(texture[:, :, 0])])
    elif texture.shape[2] == 2:
        texture = np.dstack([texture[:, :, 0], texture[:, :, 1]] + [np.zeros_like(texture[:, :, 0])] + [
            255 * np.ones_like(texture[:, :, 0])])
    elif texture.shape[2] == 3:
        texture = np.dstack(
            [texture[:, :, 0], texture[:, :, 1], texture[:, :, 2]] + [255 * np.ones_like(texture[:, :, 0])])
    elif texture.shape[2] > 4:
        texture = texture[:, :, :4]
    # Ravel width and channel count to meet pypng requirements
    texture = texture.reshape((texture.shape[0], -1))
    # Save the image
    png.from_array(texture, "RGBA8").save(output_png_file_path)


def store_noise_texture_hdr(texture, output_png_file_path, n_rank=-1):
    """This function stores the given texture to an HDR png file with 16 bits per
       channel and the specified number of channels.
      \param Texture An array of shape (Height,Width) or (Height,Width,nChannel).
             The former is handled like (Height,Width,1). It is assumed that each
             channel contains each integer value from 0 to nRank-1 exactly once. The
             range of values is remapped linearly to span the range from 0 to
             2**16-1 supported by the output format. nChannel can be 1, 2, 3 or 4.
      \param OutputPNGFilePath The path to the output *.png file including the file
             format extension.
      \param nRank Defaults to Width*Height if you pass a non-positive value."""
    # Scale the array to an HDR version
    if n_rank <= 0:
        n_rank = texture.shape[0] * texture.shape[1]
    texture = np.asarray((np.asarray(texture, dtype=np.uint64) * (2 ** 16)) // n_rank, dtype=np.uint16)
    # Get a three-dimensional array
    if len(texture.shape) < 3:
        texture = texture[:, :, np.newaxis]
    # Save the image
    Mode = ["L", "LA", "RGB", "RGBA"][texture.shape[2] - 1] + "16"
    texture = texture.reshape((texture.shape[0], -1))
    png.from_array(texture, Mode).save(output_png_file_path)


def store_nd_texture_hdr(array, output_file_path):
    """This function stores the given unsigned integer array in a minimalist binary
       file format. The last dimension is interpreted as corresponding to the
       channels of the image. The file format consists of a sequence of unsigned,
       least significant bit first 32-bit integers. The contained data is described
       below:
      - Version: File format version, should be 1.
      - nChannel: The number of color channels in the image. This should be a value
        between 1 (greyscale) and 4 (RGBA).
      - nDimension: The number of dimensions of the stored array, i.e. the number of
        indices required to uniquely identify one pixel, voxel, etc..
      - Shape[nDimension]: nDimension integers providing the size of the array along
        each dimension. By convention the first dimension is height, second width
        and third depth.
      - Data[Shape[0]*...*Shape[nDimension-1]*nChannel]: The uncompressed data of
        the array. The channels are unrolled first, followed by all dimensions in
        reverse order. Thus, an RG image of size 3*2 would be stored in the
        following order: 00R, 00G, 01R, 01G, 10R, 10G, 11R, 11G, 20R, 20G, 21R,
        21G"""
    # Prepare all the meta data and the data itself
    array = np.asarray(array, dtype=np.uint32)
    Version = 1
    nDimension = len(array.shape) - 1
    nChannel = array.shape[nDimension]
    Shape = array.shape[0:nDimension]
    Data = array.flatten("C")
    # Write it to the file
    OutputFile = open(output_file_path, "wb")
    OutputFile.write(struct.pack("LLL", Version, nChannel, nDimension))
    OutputFile.write(struct.pack("L" * nDimension, *Shape))
    OutputFile.write(struct.pack("L" * np.size(Data), *Data))
    OutputFile.close()


def load_nd_texture_hdr(source_file_path):
    """Loads a file generated by StoreNDTextureHDR() and returns it as an array like
       the one that goes into StoreNDTextureHDR() using data type np.uint32. On
       failure it returns None."""
    # Load the meta data
    file = open(source_file_path, "rb")
    Version, nChannel, nDimension = struct.unpack_from("LLL", file.read(12))
    if Version != 1:
        return None
    Shape = struct.unpack_from("L" * nDimension, file.read(4 * nDimension))
    nScalar = np.prod(Shape) * nChannel
    Data = struct.unpack_from("L" * nScalar, file.read(4 * nScalar))
    file.close()
    # Prepare the output
    return np.asarray(Data, dtype=np.uint32).reshape(tuple(list(Shape) + [nChannel]), order="C")


def generate_blue_noise_database(random_seed_index_list=range(1), min_resolution=16, max_resolution=1024,
                                 channel_count_list=[1, 2, 3, 4], standard_deviation=1.5):
    """This function generates a database of blue noise textures for all sorts of
       use cases. It includes power-of-two resolutions from MinResolution**2 up
       to MaxResolution**2. Textures are generated with each given number of
       channels. Each texture is generated multiple times using different random
       numbers per entry in RandomSeedIndexList and the entries become part of the
       file name. StandardDeviation forwards to GetVoidAndClusterBlueNoise(). The
       results are stored as LDR and HDR files to a well-organized tree of
       of directories."""
    resolution = min_resolution
    while resolution <= max_resolution:
        output_directory = f"../Data/{resolution}_{resolution}"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for n_channel in channel_count_list:
            for i in random_seed_index_list:
                texture = np.dstack(
                    [get_void_and_cluster_blue_noise((resolution, resolution), standard_deviation) for j in
                     range(n_channel)])
                ldr_format = ["LLL1", "RG01", "RGB1", "RGBA"][n_channel - 1]
                hdr_format = ["L", "LA", "RGB", "RGBA"][n_channel - 1]
                store_noise_texture_ldr(texture, os.path.join(output_directory, f"LDR_{ldr_format}_{i:d}.png"))
                store_noise_texture_hdr(texture, os.path.join(output_directory, f"HDR_{hdr_format}_{i:d}.png"))
                print(f"{resolution:d}*{resolution:d}, {ldr_format}, {i:d}")
        resolution *= 2


def generate3_d_blue_noise_texture(width, height, depth, n_channel, standard_deviation=1.5):
    """This function generates a single 3D blue noise texture with the specified
       dimensions and number of channels. It then outputs it to a sequence of Depth
       output files in LDR and HDR in a well-organized tree of directories. It also
       outputs raw binary files.
      \sa StoreNDTextureHDR() """
    output_directory = f"../Data/{width:d}_{height:d}_{depth:d}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Generate the blue noise for the various channels using multi-threading
    channel_texture_list = [None] * n_channel
    channel_thread_list = [None] * n_channel

    def generate_and_store_texture(index):
        channel_texture_list[index] = get_void_and_cluster_blue_noise((height, width, depth), standard_deviation)

    for i in range(n_channel):
        channel_thread_list[i] = threading.Thread(target=generate_and_store_texture, args=(i,))
        channel_thread_list[i].start()
    for Thread in channel_thread_list:
        Thread.join()
    texture = np.concatenate([channel_texture_list[i][:, :, :, np.newaxis] for i in range(n_channel)], 3)
    ldr_format = ["LLL1", "RG01", "RGB1", "RGBA"][n_channel - 1]
    hdr_format = ["L", "LA", "RGB", "RGBA"][n_channel - 1]
    store_nd_texture_hdr(texture, os.path.join(output_directory, "HDR_" + hdr_format + ".raw"))
    for i in range(depth):
        store_noise_texture_ldr(texture[:, :, i, :], os.path.join(output_directory, f"LDR_{ldr_format}_{i:d}.png"),
                                height * width * depth)
        store_noise_texture_hdr(texture[:, :, i, :], os.path.join(output_directory, f"HDR_{hdr_format}_{i:d}.png"),
                                height * width * depth)


def generate_nd_blue_noise_texture(shape, n_channel, output_file_path, standard_deviation=1.5):
    """This function generates a single n-dimensional blue noise texture with the
       specified shape and number of channels. It then outputs it to the specified
       raw binary file.
      \sa StoreNDTextureHDR() """
    output_directory = os.path.split(output_file_path)[0]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Generate the blue noise for the various channels using multi-threading
    channel_texture_list = [None] * n_channel
    channel_thread_list = [None] * n_channel

    def generate_and_store_texture(index):
        channel_texture_list[index] = get_void_and_cluster_blue_noise(shape, standard_deviation)

    for i in range(n_channel):
        channel_thread_list[i] = threading.Thread(target=generate_and_store_texture, args=(i,))
        channel_thread_list[i].start()
    for thread in channel_thread_list:
        thread.join()
    texture = np.concatenate([channel_texture_list[i][..., np.newaxis] for i in range(n_channel)], len(shape))
    store_nd_texture_hdr(texture, output_file_path)


def uniform_to_triangular_distribution(uniform_texture):
    """Given an array with a uniform distribution of values, this function
       constructs an array of equal shape with a triangular distribution of values.
       This is accomplished by applying a differentiable, monotonously growing
       function per entry.
      \param UniformTexture An integer array containing each value from 0 to
             np.size(UniformTexture)-1 exactly once.
      \return A floating-point array with values between -1 and 1 where the density
              grows linearly between -1 and 0 and falls linearly between 0 and 1."""
    normalized = (np.asarray(uniform_texture, dtype=float) + 0.5) / float(np.size(uniform_texture))
    return np.where(normalized < 0.5, np.sqrt(2.0 * normalized) - 1.0, 1.0 - np.sqrt(2.0 - 2.0 * normalized))


if __name__ == "__main__":
    # generate_blue_noise_database(range(64),16,64,range(1,5),1.9)
    # generate_blue_noise_database(range(16),128,128,range(1,5),1.9)
    # generate_blue_noise_database(range(8),256,256,range(1,5),1.9)
    # generate_blue_noise_database(range(1),512,512,range(1,5),1.9)
    # generate_blue_noise_database(range(1),1024,1024,[4],1.9)
    # for n_channel in range(1, 5):
    #     generate3_d_blue_noise_texture(16, 16, 16, n_channel, 1.9)
    #     generate3_d_blue_noise_texture(32, 32, 32, n_channel, 1.9)
    #     generate3_d_blue_noise_texture(64, 64, 64, n_channel, 1.9)
    #     ChannelNames=["","L","LA","RGB","RGBA"][n_channel]
    #     generate_nd_blue_noise_texture((8,8,8,8), n_channel, "../Data/8_8_8_8/HDR_" + ChannelNames + ".raw", 1.9)
    #     generate_nd_blue_noise_texture((16,16,16,16), n_channel, "../Data/16_16_16_16/HDR_" + ChannelNames + ".raw", 1.9)
    #

    texture = get_void_and_cluster_blue_noise((64, 64), 1.9)

    # texture = get_void_and_cluster_blue_noise((32, 32, 32), 1.9)[:, :, 0]
    analyze_noise_texture(texture, True)
    plot_binary_patterns(texture, 3, 5)
    pyplot.show()
