from django.urls import path
from . import views

urlpatterns = [
    path('search/', views.SearchView.as_view(), name='search'),
    path('upload/', views.UploadView.as_view(), name='upload'),
]
