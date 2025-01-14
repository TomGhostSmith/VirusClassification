import os
import zipfile

from config import config
from utils import IOUtils

def zipFiles():
    with zipfile.ZipFile(f'{config.tempFolder}/csvFiles.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        sourcePath = f"{config.dataRoot}/results"
        for root, dirs, files in os.walk(sourcePath):
            for file in files:
                if (file.endswith('.csv') or file.endswith('.xlsx')):
                    path = f"{root}/{file}"
                    zipf.write(path, os.path.relpath(path, sourcePath))
                    IOUtils.showInfo(f'Add {path} to zip file')
    
    IOUtils.showInfo(f'Stored all csv files at {config.tempFolder}/csvFiles.zip')


def main():
    zipFiles()


if (__name__ == '__main__'):
    main()  