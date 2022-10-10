import time
from deepface import DeepFace
from collect_images import collect


def run():
    response = input("Enter '1' for facial recognition or '2' to collect image sample: ")
    if response == str(1):
        DeepFace.stream('database', model_name='Facenet', detector_backend='dlib',
                        distance_metric='euclidean', time_threshold=1, frame_threshold=1)
    elif response == str(2):
        collect()
    else:
        time.sleep(1)
        print("wrong input")
        run()


if __name__ == '__main__':
    run()
