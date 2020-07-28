from django.urls import path
from . import views

app_name = "main"

urlpatterns = [
    path('', views.MyFunc.as_view(), name='func'),
    path('catdog/', views.CatDog.as_view(), name='cat_dog'),
]
