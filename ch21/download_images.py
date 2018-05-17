import argparse
import requests
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory of images')
ap.add_argument('-n', '--num-images', type=int, default=500, help='# of images to download')
args = vars(ap.parse_args())


url = 'https://www.e-zpassny.com/vector/jcaptcha.do'
total = 0

for i in range(args['num_images']):
    try:
        r = requests.get(url, timeout=60)

        # save the image to disk
        p = os.path.sep.join([args['output'], '{}.jpg'.format(str(total).zfill(5))])
        with open(p, 'wb') as fp:
            fp.write(r.content)

        print('[INFO] downloaded: {}'.format(p))
        total += 1
    except:
        print('[INFO] error downloading image ...')

    time.sleep(0.1)
