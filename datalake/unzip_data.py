from zipfile import ZipFile


def unzip_file(name_zip):

    with ZipFile(name_zip, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()


name_zip = "ADABKRW-1h-2020-08.zip"
unzip_file(name_zip)
