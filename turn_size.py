from PIL import Image

im =Image.open("pants.jpg")
out = im.resize((300, 300), Image.ANTIALIAS)
out.save('new_image.jpg')
