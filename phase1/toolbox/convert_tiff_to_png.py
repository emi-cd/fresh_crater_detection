from PIL import Image, ImageSequence, ImageOps

img = Image.open('M1132006428RC-3900-3600.tif')

for i, page in enumerate(ImageSequence.Iterator(img)):
    page = ImageOps.invert(page)
    page.save("image_file_{}.jpg".format(i), format='jpeg')