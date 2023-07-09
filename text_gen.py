from typing import Dict, List
import numpy as np
import freetype
import json
import uuid
import cv2
import os

def find_true_bb(patch: np.ndarray) -> List[int]:
    x0 = 0
    while np.sum(patch[:, x0]) == 0:
        x0 += 1
    y0 = 0
    while np.sum(patch[y0, :]) == 0:
        y0 += 1
    x1 = patch.shape[1] - 1
    while np.sum(patch[:, x1]) == 0:
        x1 -= 1
    y1 = patch.shape[0] - 1
    while np.sum(patch[y1, :]) == 0:
        y1 -= 1
    x1 += 1
    y1 += 1
    return [x0, y0, x1, y1]


def render_text(
        text: str,
        font_file: str,
        font_size: int = 60,
        spacing: int = 0
    ) -> Dict:

    face = freetype.Face(font_file)
    face.set_char_size(font_size * 64)

    text = [char for char in text if face.get_char_index(char) != 0]
    text = ''.join(text).strip()

    if len(text) == 0:
        return None

    width, height = 0, 0
    ascender, descender = 0, 0
    prev_char = None
    min_x_left = float('inf')
    for idx, char in enumerate(text):
        face.load_char(char)
        kerning = face.get_kerning(prev_char, char) if prev_char is not None else face.get_kerning(0, char)
        x_left = width + face.glyph.bitmap_left + (kerning.x >> 6) + spacing
        min_x_left = min(min_x_left, x_left)
        width += (face.glyph.advance.x >> 6) + (kerning.x >> 6) + spacing
        height = max(height, face.glyph.bitmap.rows)
        ascender = max(ascender, face.glyph.bitmap_top)
        descender = max(descender, face.glyph.bitmap.rows - face.glyph.bitmap_top)
        prev_char = char

    height = ascender + descender
    # the 10 bellow is for safety, some fonts' bitmaps are causing out of bounds error when rendering, and this fixes it
    bitmap = np.zeros((height, width - min_x_left + 10), dtype=np.ubyte)
    x, y = -min_x_left, ascender
    prev_char = None

    boxes = []
    true_width = 0
    for idx, char in enumerate(text):
        face.load_char(char)
        glyph_bitmap = face.glyph.bitmap
        kerning = face.get_kerning(prev_char, char) if prev_char is not None else face.get_kerning(0, char)

        x_left = x + face.glyph.bitmap_left + (kerning.x >> 6) + spacing
        y_top = y - face.glyph.bitmap_top
        x_right = x_left + glyph_bitmap.width
        y_bottom = y_top + glyph_bitmap.rows

        patch = np.array(glyph_bitmap.buffer, dtype=np.ubyte).reshape(glyph_bitmap.rows, glyph_bitmap.width)
        bitmap[y_top:y_bottom, x_left:x_right] = np.maximum(
            patch,
            bitmap[y_top:y_bottom, x_left:x_right]
        )
        true_bb = [0, -int(font_size*.5), (face.glyph.advance.x >> 6), int(font_size*.5)]
        if char != ' ':
            true_bb = find_true_bb(patch)
        true_width = x_left + true_bb[2] + 1
        bx = x_left + true_bb[0]
        by = y_top + true_bb[1]
        bw = true_bb[2] - true_bb[0]
        bh = true_bb[3] - true_bb[1]
        if char == ' ':
            by -= int(font_size*.5)
        boxes.append(
            {
                'char': char,
                'bounds': {'x': bx, 'y': by, 'w': bw, 'h': bh}
            }
        )
        x += (face.glyph.advance.x >> 6) + (kerning.x >> 6) + spacing
        prev_char = char

    bitmap = bitmap[:, :true_width]
    bbox = find_true_bb(bitmap)

    bitmap = bitmap[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    true_width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    for box in boxes:
        box['bounds']['x'] = (box['bounds']['x']) / true_width
        box['bounds']['y'] = (box['bounds']['y']) / height
        box['bounds']['w'] = (box['bounds']['w']) / true_width
        box['bounds']['h'] = (box['bounds']['h']) / height

    return {
        'image': bitmap/255.,
        'boxes': boxes
    }


def demo() -> None:
    text = "hello world xyzq fjq.:-+129?"
    font_file = '../ttfs/helvetica.ttf'

    result = render_text(
        text=text,
        font_file=font_file,
        font_size=90,
        spacing=0,
    )
    if not result:
        print(f'nothing generated with {font_file}')
        return
    
    img = result['image']
    boxes = result['boxes']

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    id = str(uuid.uuid4())
    font_name = os.path.basename(font_file).split('.')[0]
    image_path = f'{output_dir}/{id}.png'
    json_path = f'{output_dir}/{id}.json'
    
    img = (result['image']*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(image_path, img)
    annotation = {
        'segment': {
            'id': id,
            'width': img.shape[1],
            'height': img.shape[0],
            'rects': [box['bounds'] for box in boxes],
            'decChars': [ord(box['char']) for box in boxes],
            'transcribed': ''.join([box['char'] for box in boxes]),
        },
        'font': font_name,
    }
    with open(json_path, 'w') as f:
        f.write(json.dumps(annotation, indent=4))


if __name__ == "__main__":
    demo()