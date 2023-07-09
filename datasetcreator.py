from text_gen import render_text
import cv2
import uuid
import json
import os
import numpy as np
import glob
import string
import shutil
import tqdm
import argparse
import albumentations as A

from faker import Faker
fake = Faker()
locales = ['en_US', 'es_ES', 'fr_FR']
fonts = [f.split('/')[-1].split('.')[0] for f in glob.glob('ttfs2/*.ttf')]

import multiprocessing as mpi

parser = argparse.ArgumentParser(description='Font data generator')
parser.add_argument('output_folder', default='test', type=str, help='Preview training data samples')
parser.add_argument('samples', default=10, type=int, help='Preview training data samples')
args = parser.parse_args()

total = args.samples*len(fonts)

if os.path.exists(args.output_folder):
    shutil.rmtree(args.output_folder)
os.makedirs(args.output_folder)

bgimage = cv2.imread('background.png', 0)/255.

letters = string.ascii_lowercase
letters += string.ascii_uppercase
letters += string.ascii_letters
letters += string.digits
letters += string.punctuation
letters = list(letters)

def random_word(length=np.random.randint(1, 10)):
    length = int(1 + 14*np.power(np.random.random(), .6))
    txt = ''.join(letters[np.random.randint(0, len(letters))] for _ in range(length))
    ntxt = ''
    for c in txt:
        ntxt += c
        if np.random.random() < .14:
            ntxt += ' '
    ntxt = ntxt.strip()

    if np.random.random() < .9:
        chc = np.random.randint(0, 4)
        if chc == 0:
            ntxt = fake.name()
            words = ntxt.split(' ')
            np.random.shuffle(words)
            ntxt = ' '.join(words)
        elif chc == 1:
            ntxt = fake.address()
            words = ntxt.split(' ')
            np.random.shuffle(words)
            ntxt = ' '.join(words)
        elif chc == 2:
            ntxt = fake.text(max_nb_chars=16)
            words = ntxt.split(' ')
            np.random.shuffle(words)
            ntxt = ' '.join(words)
        elif chc == 3:
            ntxt = fake.date()
            nums = ntxt.split('-')
            np.random.shuffle(nums)
            deliminator = np.random.choice(['-', '/', '.'])
            ntxt = deliminator.join(nums)

    if '\n' in ntxt:
        ntxt = ntxt.split('\n')
        ntxt = ntxt[np.random.randint(0, len(ntxt))]

    return ntxt[:18]


def augment(image):

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    image = (.5 + .5*np.random.random())*image

    h, w = image.shape[:2]
    left_padding = np.random.randint(0, 20)
    right_padding = np.random.randint(0, 20)
    top_padding = np.random.randint(0, 20)
    bottom_padding = np.random.randint(0, 20)

    image = np.hstack([np.zeros((h, left_padding)), image, np.zeros((h, right_padding))])
    image = np.vstack([np.zeros((top_padding, w+left_padding+right_padding)), image, np.zeros((bottom_padding, w+left_padding+right_padding))])

    random_rotation = np.random.randint(-2, 2)

    image = rotate_image(image, random_rotation)

    elastic1 = A.ElasticTransform(alpha=2, sigma=1, alpha_affine=0, p=.75)
    elastic2 = A.ElasticTransform(alpha=12, sigma=12, alpha_affine=0, p=.5)

    image = elastic1(image=image)['image']
    image = elastic2(image=image)['image']

    # blur
    blur1 = A.Blur(blur_limit=3, p=.75)
    image = blur1(image=image)['image']
    image = elastic1(image=image, p=.5)['image']
    blur2 = A.Blur(blur_limit=3, p=.75)
    image = blur2(image=image)['image']

    h, w = image.shape[:2]
    rx = np.random.randint(0, bgimage.shape[0] - h)
    ry = np.random.randint(0, bgimage.shape[1] - w)
    bg_crop = bgimage[rx:rx+h, ry:ry+w]
    bg_crop = .5 + np.random.random()*.4*bg_crop

    image = 1. - image

    result = image * bg_crop

    if np.random.random() < .1:
        result = 1. - result

    # contrast
    contrast1 = A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, always_apply=False, p=1)
    result = contrast1(image=(result*255).astype(np.uint8))['image']/255

    # random noise
    noise1 = A.GaussNoise(var_limit=(2.6, 24.7), mean=0, always_apply=True, p=1)
    result = noise1(image=(result*255).astype(np.uint8))['image']/255

    ccc = cv2.Canny((result*255).astype(np.uint8), 100, 200)/255.
    if np.sum(ccc) == 0:
        return None

    return result

def generate(*qargs):
    idx = qargs[0]
    np.random.seed(idx)
    global fake
    fake = Faker(locales[np.random.randint(0, len(locales))])
    Faker.seed(np.random.randint(0, 100000))
    font = fonts[np.random.randint(0, len(fonts))]
    fontpath = 'ttfs2/' + font + '.ttf'
    text = random_word()
    if not os.path.exists(fontpath):
        print('font not found')
        print(font)
        return
    
    try:
        render_result = render_text(text, fontpath, font_size=80, spacing=np.random.randint(-3, 3))
    except:
        print('????????????')
        return

    try:
        assert render_result['image'] is not None
    except:
        return
    
    image = render_result['image']
    cv2.imwrite('kaj.png', (image*255))

    image = augment(image)
    if image is None:
        return
    outname = str(uuid.uuid4())

    info = {
        'transcribed': text,
        'font': font
    }

    cv2.imwrite(f'{args.output_folder}/{outname}.png', (255*image).astype(int))
    with open(f'{args.output_folder}/{outname}.json', 'w') as f:
        f.write(json.dumps(info, indent=4))

def main():

    if True:
        with mpi.Pool(mpi.cpu_count()) as p:
            with tqdm.tqdm(total=total) as pbar:
                for _ in p.imap_unordered(generate, [idx for idx in range(0, total)]):
                    pbar.update()
    else:
        for idx in tqdm.tqdm(range(0, total)):
            generate(idx)

   
if __name__ == '__main__':
    main()