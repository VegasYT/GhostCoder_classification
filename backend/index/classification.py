import zipfile
from ai.main import distribution
import io

def create_zip_with_folders(files, output_zip):
    output_zip = io.BytesIO()
    
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in files:
            with open(f'{file["file"]}', 'rb') as f:
                zipf.writestr(f"{file['label']}/{file['file_name']}", f.read())
    
    return output_zip