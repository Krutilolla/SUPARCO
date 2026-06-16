
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Sudhanshu:Sudhanshu@cluster0.uj38l.mongodb.net/?appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client.session
collection = db["sessions"]
