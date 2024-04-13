# models.py
from django.db import models


class UploadedFile(models.Model):
    file_name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploaded_files/')  # Путь для сохранения файлов

    def __str__(self):
        return self.file_name