import numpy as np
import os

class FaceCamDataManager(object) :
    __slots__ = ['time_stamps', 'fs', 'facemotion', 'pupil']
    
    def __init__(self, base_path):
        self.time_stamps = self.get_time_stamps(base_path)
        self.fs = 1. / np.mean(self.time_stamps[1:] - self.time_stamps[:-1])
        self.facemotion, self.pupil = self.get_face_metrics(base_path)

    def __str__(self) :
        list_attr = ['time_stamps', 'fs', 'facemotion', 'pupil']
        return f"Available attributes : {list_attr}"
    
    def get_time_stamps(self, base_path):
        face_camera = np.load(os.path.join(base_path,"FaceCamera-summary.npy"), allow_pickle=True)
        return face_camera.item().get('times')
    
    def get_face_metrics(self, base_path):
        face_it_path = os.path.join(base_path, "FaceIt", "FaceIt.npz")
        faceitOutput = np.load(face_it_path, allow_pickle=True)
        facemotion = (faceitOutput['motion_energy'])
        pupil = (faceitOutput['pupil_dilation'])
        return facemotion, pupil

if __name__ == "__main__":
    base_path = "Y:/raw-imaging/TESTS/Mai-An/visual_test/16-00-59"
    face_cam = FaceCamDataManager(base_path)
    print(face_cam)
