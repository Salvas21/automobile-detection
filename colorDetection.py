def detectColor(img, x, y, w, h):
    test = img[y + round((h / 2)), x + round((w / 2))]
    color = test.astype(int)
    return color


def find_most_common_pixel(image):
    histogram = {}  # Dictionary keeps count of different kinds of pixels in image
    for row in image:
        for pixel in row:
            # pixel_val = get_pixel_value(pixel)

            pixel_val = rgb2int(pixel)
            if pixel_val in histogram:
                histogram[pixel_val] += 1  # Increment count
            else:
                histogram[pixel_val] = 1  # pixel_val encountered for the first time

    mode_pixel_val = max(histogram, key=histogram.get)  # Find pixel_val whose count is maximum
    # print(int2rgb(mode_pixel_val))
    return int2rgb(mode_pixel_val)


def rgb2int(rgb): return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


def int2rgb(N):   return N >> 16, (N >> 8) % 256, N % 256


def get_rgb_values(pixel_value):
    red = pixel_value % 256
    pixel_value //= 256
    green = pixel_value % 256
    pixel_value //= 256
    blue = pixel_value
    return [red, green, blue]


def get_pixel_value(pixel):
    return pixel[0] + 256 * pixel[1] + 256 * 256 * pixel[2]
