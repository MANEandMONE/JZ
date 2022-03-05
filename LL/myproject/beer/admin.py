from django.contrib import admin
from .models import *
# from .models import User

# Register your models here.


class HotelAdmin(admin.ModelAdmin):
    list_display = ('index', 'locate', 'name', 'rating', 'review',
                    'classfications', 'address', 'cost', 'url')


class RestaurantAdmin(admin.ModelAdmin):
    list_display = ('index', 'locate', 'name', 'rating', 'classfications',
                    'address', 'hour', 'url')


class ResultAdmin(admin.ModelAdmin):
    list_displat = ('locate', 'rating', 'review', 'rating', 'classfications',
                    'address', 'explain', 'mood', 'topic', 'reason', 'Cluster')


admin.site.register(Hotel, HotelAdmin)

admin.site.register(Restaurant, RestaurantAdmin)

admin.site.register(Result, ResultAdmin)
# class UserAdmin(admin.ModelAdmin) :
#     list_display = ('username', 'password')

# admin.site.register(User, UserAdmin)