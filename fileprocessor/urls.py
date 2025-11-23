from django.urls import path
from .views import file_process_view

urlpatterns = [
    path('', file_process_view, name='file_process'),
]