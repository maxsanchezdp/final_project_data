import os

def play_song(path):
    os.system("mpg123 " + path)
