# Generated by Django 4.2 on 2025-03-12 07:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0011_alter_game_header_image_alter_game_trailer_url_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='email_verification_token',
        ),
    ]
