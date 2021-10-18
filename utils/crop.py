def is_in_box(big_box, small_box):
    """
    Determine whether the label box is in the sliding window

    :param big_box: sliding window box [x, y, w, h]
    :param small_box: label box [x, y, w, h]

    """
    x1, y1, w1, h1 = big_box
    x2, y2, w2, h2 = small_box
    if x2 >= x1:
        if y2 >= y1:
            if x2+w2 <= x1+w1:
                if y2+h2 <= y1+h1:
                    return True
    
    return False

def crop_window(image, bbox):
    """
    Crop specific area in the image
    """
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    return image[y: y+h, x: x+w]

def anno_convert(window_box, label_box):
    wx, wy = max(0, window_box[0]), max(0, window_box[1])
    lx, ly, lw, lh = label_box
    lx -= wx
    ly -= wy
    return [lx, ly, lw, lh]