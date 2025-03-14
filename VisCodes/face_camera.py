import numpy as np
import os
import datetime

class FaceCamDataManager(object) :
    __slots__ = ['time_stamps', 'fs', 'facemotion', 'pupil']
    
    def __init__(self, base_path, timestamp_start):
        self.time_stamps = self.get_time_stamps(base_path, timestamp_start)
        self.fs = 1. / np.mean(self.time_stamps[1:] - self.time_stamps[:-1])
        self.facemotion, self.pupil = self.get_face_metrics(base_path)

    def __str__(self) :
        list_attr = ['time_stamps', 'fs', 'facemotion', 'pupil']
        return f"Available attributes : {list_attr}"
    
    def load_FaceCamera_data(self, imgfolder, t0=0, verbose=True,
                             produce_FaceCamera_summary=True, N_summary=5):
        '''
        From https://github.com/yzerlaut/physion/blob/main/src/physion/assembling/tools.py
        '''
        file_list = [f for f in os.listdir(imgfolder) if f.endswith('.npy')]
        _times = np.array([float(f.replace('.npy', '')) for f in file_list])
        _isorted = np.argsort(_times)
        times = _times[_isorted]-t0
        FILES = np.array(file_list)[_isorted]
        nframes = len(times)
        Lx, Ly = np.load(os.path.join(imgfolder, FILES[0])).shape
        if verbose:
            print('Sampling frequency: %.1f Hz  (datafile: %s)' % (1./np.diff(times).mean(), imgfolder))
            
        if produce_FaceCamera_summary:
            fn = os.path.join(imgfolder, '..', 'FaceCamera-summary.npy')
            data = {'times':times, 'nframes':nframes, 'Lx':Lx, 'Ly':Ly, 'sample_frames':[], 'sample_frames_index':[]}
            for i in np.linspace(0, nframes-1, N_summary, dtype=int):
                data['sample_frames'].append(np.load(os.path.join(imgfolder, FILES[i])))
                data['sample_frames_index'].append(i)
            np.save(fn, data)
            
        return times, FILES, nframes, Lx, Ly

    def get_time_stamps(self, base_path, timestamp_start):
        try :
            face_camera = np.load(os.path.join(base_path, "FaceCamera-summary.npy"), allow_pickle=True)
        except :
            print("No FaceCamera-summary.npy file found. Creating one.")
            face_cam_img_path = os.path.join(base_path, "FaceCamera-imgs")
            self.load_FaceCamera_data(face_cam_img_path)
            face_camera = np.load(os.path.join(base_path, "FaceCamera-summary.npy"), allow_pickle=True)
        
        if face_camera.item().get('times')[0] > 1e8 :
            times = np.array([(datetime.datetime.fromtimestamp(t) - timestamp_start).total_seconds() for t in face_camera.item().get('times')])
        else :
            times = face_camera.item().get('times')

        return times
    
    def get_face_metrics(self, base_path):
        face_it_path = os.path.join(base_path, "FaceCamera-imgs", "FaceIt", "FaceIt.npz")
        if not os.path.exists(face_it_path) :
            face_it_path = os.path.join(base_path, "FaceIt", "FaceIt.npz")
            if not os.path.exists(face_it_path) :
                raise Exception(f"{face_it_path} doesn't exist.")
        
        faceitOutput = np.load(face_it_path, allow_pickle=True)
        facemotion = (faceitOutput['motion_energy'])
        pupil = (faceitOutput['pupil_dilation'])
        return facemotion, pupil
            

if __name__ == "__main__":
    import Photodiode

    base_path = "Y:/raw-imaging/Adrianna/experiments/NDNF/2025_02_26/12-28-29"
    timestamp_start = Photodiode.get_timestamp_start(base_path)
    face_cam = FaceCamDataManager(base_path, timestamp_start)
    print(face_cam)
