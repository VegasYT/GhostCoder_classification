from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .models import UploadedFile
from django.views.decorators.csrf import csrf_exempt
import uuid
from .converters import convert_to_txt
import pythoncom
import os
import json
from collections import defaultdict

from ai.main import distribution
from .classification import create_zip_with_folders
from .csv_merge import merge_csv
from ai.learn import train


@csrf_exempt
def upload_file_view(request):
    if request.method == 'POST' and request.FILES:
        uploaded_files = request.FILES.getlist('files')
        hierarchy = bool(request.POST.get('hierarchy'))
        files = []
        label_counts = defaultdict(int)
        errors = []

        with open('index/class_counts.json', 'r', encoding='utf-8') as json_file:
            class_count = json.load(json_file)

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

            label_counts[label] += 1

        # Проверка количества файлов каждого класса
        for label, count in class_count.items():
            if label_counts[label] < count:
                errors.append(f'Недостаточно файлов класса {label}: {count-label_counts[label]}')

        
        zip_archive = create_zip_with_folders(files=files, output_zip='sorted.zip')
        zip_archive.seek(0)
        
        response = HttpResponse(zip_archive, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="sorted.zip"'

        response['X-Error-Message'] = json.dumps({'errors': errors})

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
            os.remove(uploaded_csv_path)
            train()

            return JsonResponse({'success': 'csv файл загружен'})

        else:
            return JsonResponse({'error': 'Неподдерживаемый формат файла. Ожидается файл с расширением .csv'}, status=400)

    else:
        return JsonResponse({'error': 'Некорректный запрос'}, status=400)


@csrf_exempt
def get_class_counters(request):
    if request.method == 'GET':
        try:
            with open('index/class_counts.json', 'r', encoding='utf-8') as json_file:
                class_counters = json.load(json_file)
            return JsonResponse(class_counters)
        except FileNotFoundError:
            return JsonResponse({'error': 'Файл class_counters.json не найден'}, status=404)
    else:
        return JsonResponse({'error': 'Метод не поддерживается'}, status=405)
    

@csrf_exempt
def save_class_counts(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))

            # Открываем файл class_counts.json и загружаем данные
            with open('index/class_counts.json', 'r', encoding='utf-8') as json_file:
                class_counts = json.load(json_file)

            # Обновляем значения в соответствии с данными из запроса
            for key, value in data.items():
                print(f'{key} - {value}')
                class_counts[key] = value

            # Сохраняем обновленные данные в файл class_counts.json
            with open('index/class_counts.json', 'w', encoding='utf-8') as json_file:
                json.dump(class_counts, json_file, ensure_ascii=False, indent=4)

            return JsonResponse({'success': 'Данные сохранены в class_counts.json'})
        except Exception as e:
            return JsonResponse({'error': f'Произошла ошибка при сохранении данных: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Некорректный запрос'}, status=400)