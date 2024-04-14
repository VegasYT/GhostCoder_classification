from django.urls import path
from index import views
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload_file/', views.upload_file_view, name='upload_file'),
    path('upload_csv/', views.upload_csv_view, name='upload_csv'),
    path('get_classes/', views.get_class_counters, name='get_classes'),
    path('save_class_counts/', views.save_class_counts, name='save_class_counts')
]
