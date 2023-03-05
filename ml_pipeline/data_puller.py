import os
from pathlib import Path
import shutil

from zenml.steps import step
from mongodb.utils import get_db

BASE_DIR = "/home/samjoel/Projects/mlops_on_edge/data"

@step
def data_puller(db_name: str, save_dir:str) -> str:
    db = get_db(db_name)
    save_path = os.path.join(BASE_DIR, save_dir)
    for data in db.data.find():
        label = data["label"]
        if label:
            image_dir = os.path.join(save_path, label)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir, mode = 666, exist_ok=True)
            image_name = Path(data["image_path"]).name
            shutil.copy(data["image_path"], os.path.join(image_dir, image_name))

    return save_path
        

    
    