def find_most_common_pixel(image):
    histogram = {}  # Keeps count of different kinds of pixels in image
    for row in image:
        for pixel in row:
            pixel_val = rgb2int(pixel)  # Convert rgb to int
            if pixel_val in histogram:
                histogram[pixel_val] += 1  # Increment count
            else:
                histogram[pixel_val] = 1  # pixel_val encountered for the first time

    mode_pixel_val = max(histogram, key=histogram.get)  # Find pixel_val whose count is maximum
    return int2rgb(mode_pixel_val)


def rgb2int(rgb): return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


def int2rgb(n): return n >> 16, (n >> 8) % 256, n % 256
