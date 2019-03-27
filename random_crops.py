import random, os, time, math
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

INPATH = dir_path + "/test_img"
OUTPATH = dir_path + "/cropped"



crop_percentage = .50


tilesPerImage = 100

files = os.listdir(INPATH)
numOfImages = len(files)

t = time.time()
for file in files:
   with Image.open(os.path.join(INPATH, file)) as im:
     for i in range(1, tilesPerImage+1):
       newname = file.replace('.', '_{:03d}.'.format(i))
       w, h = im.size
       dx = dy = math.ceil(min(w,h)*crop_percentage)
       x = random.randint(0, w-dx-1)
       y = random.randint(0, h-dy-1)
       print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
       im.crop((x,y, x+dx, y+dy))\
         .save(os.path.join(OUTPATH, newname))

t = time.time()-t
print("Done {} images in {:.2f}s".format(numOfImages, t))
print("({:.1f} images per second)".format(numOfImages/t))
print("({:.1f} tiles per second)".format(tilesPerImage*numOfImages/t))