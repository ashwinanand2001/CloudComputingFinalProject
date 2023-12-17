from django.db import models

class Movie(models.Model):
    title = models.CharFiels(max_length=255)

class UserSearch(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    search_query = models.CharField(max_length=255)
