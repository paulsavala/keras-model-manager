from pathlib import Path


def delete_folder(path) :
    for sub in path.iterdir() :
        if sub.is_dir() :
            delete_folder(sub)
        else:
            sub.unlink()
    path.rmdir()
