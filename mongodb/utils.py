from pymongo import MongoClient


def get_db(db_name):
    client = MongoClient("localhost", 27017, username="root", password="root")
    db = client[db_name]
    return db
