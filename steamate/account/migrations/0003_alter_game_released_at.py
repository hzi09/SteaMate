# Generated by Django 4.2 on 2025-03-09 07:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0002_alter_user_steam_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='game',
            name='released_at',
            field=models.DateField(blank=True, null=True),
        ),
    ]
