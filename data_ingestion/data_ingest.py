import paho.mqtt.client as mqtt
from uuid import uuid4
from mqtt_broker.constants import MQTT_PATH, MQTT_SERVER, MQTT_PORT
from mongodb.utils import get_db


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_PATH)
    # The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
    image_name = f"{uuid4()}.jpg"
    image_path = f"/home/samjoel/Projects/mlops_on_edge/data/raw_data/{image_name}"
    with open(image_path, "wb") as f:
        db = get_db("mlops-raw")
        db.data.insert_one({"image_path": image_path, "label": None})
        f.write(msg.payload)
        print("Image Received")
        f.close()


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_SERVER, MQTT_PORT, 60)


client.loop_forever()
