# https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
import os
import sys
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                ziph.write(os.path.join(root, file))

if __name__ == '__main__':
    dest = sys.argv[1]
    src = sys.argv[2]
    zipf = zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED)
    zipdir(src, zipf)
    zipf.close()
