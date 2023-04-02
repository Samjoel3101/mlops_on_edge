from pathlib import Path
from mongodb.utils import get_db


def upload_dataset_labels(db_name, image_dir):
    db = get_db(db_name)
    for dir in Path(image_dir).iterdir():
        for image_path in dir.iterdir():
            db.data.insert_one({"image_path": str(image_path), "label": dir.name})


if __name__ == "__main__":
    upload_dataset_labels("mlops-new", "/home/samjoel/Projects/mlops_on_edge/data/new_dataset")