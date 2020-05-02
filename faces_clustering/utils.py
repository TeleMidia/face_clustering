import os


def is_image(file):
    return file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


def get_files_folder(folder, criteria=lambda x: True):
    complete_urls = []
    for a, b, c in os.walk(folder):
        complete_urls += [os.path.join(a, x) for x in c if criteria(x)]
    return complete_urls
