from zipfile import ZipFile
import splitfolders
import base64

def unzipFile(filePath):
    with ZipFile(filePath, 'r') as zip:
        zip.extractall()

def splitTrainTestValidFolders(folderName):
    splitfolders.ratio(folderName, output='output', seed=1337, ratio=(0.8, 0.1, 0.1))

def decodeImage(imageString, fileName):
    imgdata = base64.b64decode(imageString)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImage(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())
