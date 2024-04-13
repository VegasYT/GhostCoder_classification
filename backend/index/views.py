from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .models import UploadedFile
from django.views.decorators.csrf import csrf_exempt
import uuid
from .converters import convert_to_txt
import pythoncom
import os

from ai.main import distribution
from .classification import create_zip_with_folders
from .csv_merge import merge_csv


@csrf_exempt
def upload_file_view(request):
    if request.method == 'POST' and request.FILES:
        uploaded_files = request.FILES.getlist('files')
        files = []

        for uploaded_file in uploaded_files:
            unique_filename = uploaded_file.name
            uploaded_file.name = f"{uuid.uuid4()}_{uploaded_file.name}"

            uploaded_file_instance = UploadedFile.objects.create(
                file_name=unique_filename, 
                file=uploaded_file
            )
            uploaded_file_instance.save()
            file_path = f"{uploaded_file_instance.file.name}"

            pythoncom.CoInitialize()

            text = convert_to_txt(file_path)
            label = distribution(text)
            files.append({
                'file_name': unique_filename,
                'file': uploaded_file_instance.file.name,
                'label': label
            })
        
        zip_archive = create_zip_with_folders(files=files, output_zip='sorted.zip')
        zip_archive.seek(0)
        
        response = HttpResponse(zip_archive, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="sorted.zip"'
        return response
    
    else:
        return JsonResponse({'error': 'Некорректный запрос'}, status=400)


@csrf_exempt
def upload_csv_view(request):
    if request.method == 'POST' and request.FILES:
        uploaded_csv = request.FILES.get('csv_file')

        if uploaded_csv.name.endswith('.csv'):
            uploaded_csv.name = f"{uuid.uuid4()}_{uploaded_csv.name}"

            with open(os.path.join('ai', uploaded_csv.name), 'wb') as destination:
                for chunk in uploaded_csv.chunks():
                    destination.write(chunk)

            uploaded_csv_path = os.path.join('ai', uploaded_csv.name)

            file1 = 'ai/dataset.csv'
            file2 = uploaded_csv_path
            output_file = 'ai/dataset.csv'
            merge_csv(file1, file2, output_file)

            return JsonResponse({'success': 'csv файл загружен'})

        else:
            return JsonResponse({'error': 'Неподдерживаемый формат файла. Ожидается файл с расширением .csv'}, status=400)

    else:
        return JsonResponse({'error': 'Некорректный запрос'}, status=400)