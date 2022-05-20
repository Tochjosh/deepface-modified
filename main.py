from deepface import DeepFace
from collect_images import collect

# collect()
real = DeepFace.stream('database', model_name='Facenet', detector_backend='dlib',
                       distance_metric='euclidean', time_threshold=1, frame_threshold=1)
