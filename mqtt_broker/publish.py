import paho.mqtt.publish as publish

from .constants import MQTT_PATH, MQTT_SERVER, MQTT_PORT


def publish_image(byteArr):
    publish.single(MQTT_PATH, byteArr, hostname=MQTT_SERVER, port=MQTT_PORT)


if __name__ == "__main__":
    f = open("/home/samjoel/Projects/mlops_on_edge/mqtt_broker/data/94644be7-8080-408f-aca5-21e25724a64d.png", "rb")
    fileContent = f.read()
    byteArr = bytearray(fileContent)
    publish_image(byteArr)
