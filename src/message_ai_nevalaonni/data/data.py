from extractor import Extractor
import logging
import json

if __name__ == "__main__":
    e = Extractor(r"C:\Users\nevalaonni\Downloads")

    class Data:
        def __init__(self):
            pass

    e.loop_over_folders()

    with open("messages.txt","w") as f:
        json.dump(e.get_messages(),f)


