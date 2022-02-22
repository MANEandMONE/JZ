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


admin.site.register(Hotel, HotelAdmin)

admin.site.register(Restaurant, RestaurantAdmin)

# class UserAdmin(admin.ModelAdmin) :
#     list_display = ('username', 'password')

# admin.site.register(User, UserAdmin)