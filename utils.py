import sys
import time
from PIL import Image, ImageDraw
import numpy as np

term_width = 0
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def render_grid_image(self, grid_np: np.ndarray, cell_size: int = 24) -> Image.Image:
    '''
    Render and save a single integer grid to an RGB image.
    '''
    palette = [
        (0, 0, 0),        # 0 black
        (0, 0, 255),      # 1 blue
        (255, 0, 0),      # 2 red
        (0, 255, 0),      # 3 green
        (255, 255, 0),    # 4 yellow
        (128, 128, 128),  # 5 gray
        (255, 0, 255),    # 6 magenta
        (255, 165, 0),    # 7 orange
        (0, 255, 255),    # 8 cyan
        (255, 255, 255),  # 9 white
    ]

    h, w = int(grid_np.shape[0]), int(grid_np.shape[1])
    img_w, img_h = w * cell_size, h * cell_size
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for y in range(h):
        for x in range(w):
            color = palette[int(grid_np[y, x])]
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=color)

    for x in range(1, w):
        X = x * cell_size
        draw.line([(X, 0), (X, img_h - 1)], fill=(0, 0, 0), width=1)
    for y in range(1, h):
        Y = y * cell_size
        draw.line([(0, Y), (img_w - 1, Y)], fill=(0, 0, 0), width=1)
    draw.line([(0, 0), (0, img_h - 1)], fill=(0, 0, 0), width=1)
    draw.line([(img_w - 1, 0), (img_w - 1, img_h - 1)], fill=(0, 0, 0), width=1)
    draw.line([(0, 0), (img_w - 1, 0)], fill=(0, 0, 0), width=1)
    draw.line([(0, img_h - 1), (img_w - 1, img_h - 1)], fill=(0, 0, 0), width=1)

    return canvas