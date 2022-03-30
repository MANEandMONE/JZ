from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'), )


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'), )


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'), )


class AuthUserUserPermissions(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'), )


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType',
                                     models.DO_NOTHING,
                                     blank=True,
                                     null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'), )


class DjangoMigrations(models.Model):
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Hotel(models.Model):
    index = models.IntegerField(primary_key=True)
    locate = models.TextField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    review = models.TextField(blank=True, null=True)
    classfications = models.TextField(blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    cost = models.TextField(blank=True, null=True)
    url = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'hotel'


class Restaurant(models.Model):
    index = models.IntegerField(primary_key=True)
    locate = models.TextField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    classfications = models.TextField(blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    hour = models.TextField(blank=True, null=True)
    url = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'restaurant'


class Result(models.Model):
    index = models.IntegerField(blank=True, null=True)
    locate = models.CharField(primary_key=True, max_length=45)
    rating = models.FloatField(blank=True, null=True)
    review = models.TextField(blank=True, null=True)
    classfications = models.TextField(blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    explain = models.TextField(blank=True, null=True)
    mood = models.TextField(blank=True, null=True)
    topic = models.TextField(blank=True, null=True)
    reason = models.TextField(blank=True, null=True)
    cluster = models.IntegerField(db_column='Cluster', blank=True,
                                  null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'result'


# class User(models.Model):
#     user_id = models.CharField(primary_key=True,
#                                max_length=32,
#                                unique=True,
#                                verbose_name='유저 아이디')
#     user_pw = models.CharField(max_length=128, verbose_name='유저 비밀번호')
#     user_name = models.CharField(max_length=16,
#                                  unique=True,
#                                  verbose_name='유저 이름')
#     user_email = models.EmailField(max_length=32,
#                                    unique=True,
#                                    verbose_name='유저 이메일')
#     user_register_dttm = models.DateTimeField(auto_now_add=True,
#                                               verbose_name='계정 생성시간')

#     def __str__(self):
#         return self.user_name

#     class Meta:
#         db_table = 'user'
#         verbose_name = '유저'
#         verbose_name_plural = '유저'
