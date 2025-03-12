# Generated by Django 4.2 on 2025-03-08 00:46

from django.conf import settings
import django.contrib.auth.models
import django.contrib.auth.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('username', models.CharField(error_messages={'unique': 'A user with that username already exists.'}, help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.', max_length=150, unique=True, validators=[django.contrib.auth.validators.UnicodeUsernameValidator()], verbose_name='username')),
                ('first_name', models.CharField(blank=True, max_length=150, verbose_name='first name')),
                ('last_name', models.CharField(blank=True, max_length=150, verbose_name='last name')),
                ('is_staff', models.BooleanField(default=False, help_text='Designates whether the user can log into this admin site.', verbose_name='staff status')),
                ('is_active', models.BooleanField(default=True, help_text='Designates whether this user should be treated as active. Unselect this instead of deleting accounts.', verbose_name='active')),
                ('date_joined', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date joined')),
                ('nickname', models.CharField(max_length=50)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('profile_image', models.ImageField(default='user/profile_image/user_default_image.png', upload_to='user/profile_image/')),
                ('gender', models.IntegerField(choices=[(1, '남'), (2, '여'), (3, '공개 안 함')], default=3)),
                ('birth', models.DateField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('steam_id', models.CharField(blank=True, max_length=20)),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.group', verbose_name='groups')),
            ],
            options={
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
                'abstract': False,
            },
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
        migrations.CreateModel(
            name='Game',
            fields=[
                ('appid', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=100)),
                ('genre', models.CharField(max_length=50)),
                ('released_at', models.DateField(blank=True)),
                ('description', models.TextField(blank=True)),
                ('review_score', models.FloatField(blank=True, default=0)),
                ('comment', models.TextField(blank=True)),
                ('header_image', models.TextField(blank=True)),
                ('trailer_url', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='Genre',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('genre_name', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='UserPreferredGame',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('playtime', models.IntegerField(default=0)),
                ('game', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='account.game')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'unique_together': {('user', 'game')},
            },
        ),
        migrations.AddField(
            model_name='user',
            name='preferred_game',
            field=models.ManyToManyField(blank=True, related_name='preferred_game', through='account.UserPreferredGame', to='account.game'),
        ),
        migrations.AddField(
            model_name='user',
            name='preferred_genre',
            field=models.ManyToManyField(blank=True, related_name='preferred_genre', to='account.genre'),
        ),
        migrations.AddField(
            model_name='user',
            name='user_permissions',
            field=models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.permission', verbose_name='user permissions'),
        ),
    ]
