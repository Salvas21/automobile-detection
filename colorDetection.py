def detectColor(img, x, y, w, h):
    test = img[y + round((h / 2)), x + round((w / 2))]
    color = test.astype(int)
    return color