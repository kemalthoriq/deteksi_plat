from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('results/<int:image_id>/', views.results, name='results'),
]
