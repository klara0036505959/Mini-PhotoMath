
import os
path = "C:\\Users\\klara\\Desktop\\extracted_images\\times"
counter = 0
for filename in os.listdir(path):
    if not counter%3 == 0:
        os.remove(path + '\\' + filename)
    counter += 1