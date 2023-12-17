from django.shortcuts import render
from .modelss import Movie

def recommend_system_movies(request,search_query):
    recommendations = your_recommendation_function(search_query)
    return render(request,'recommendations.html',{'reecommendations':recommendations})
# Create your views here.
