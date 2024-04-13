from django.urls import path
from index import views
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload_file/', views.upload_file_view, name='upload_file'),
    path('upload_csv/', views.upload_csv_view, name='upload_csv'),
]
