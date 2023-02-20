from moviepy.editor import ImageSequenceClip
from tensorboardX import SummaryWriter
from csv import DictWriter
from pathlib import Path

import numpy as np
import os


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

        # Homework submission
        self.log_file_path = Path(log_dir, 'log_file.csv')
        self.log_video_path = Path(log_dir, 'videos')
        self.log_video_path.mkdir(parents=True, exist_ok=True)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)
    
    def log_file(self, itr, logs):
        """Write out the logs to files for testing"""
        logs['itr'] = itr
        with open(self.log_file_path, 'a') as f:
            dictwriter_object = DictWriter(f, fieldnames=sorted(list(logs.keys())))
            if os.path.getsize(self.log_file_path) == 0:  # check if the file is empty
                dictwriter_object.writeheader()  # write the header row
            dictwriter_object.writerow(logs)
            f.close()

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_video_file(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"   
        # only record the last rollout
        video_frames  = video_frames.transpose(0, 1, 3, 4, 2)
        for video_idx in range(len(video_frames)):
            clip = ImageSequenceClip(list(video_frames[video_idx]), fps=fps)
            clip_name = f'{name}_step_{step}_video{video_idx+1}.gif'
            clip.write_gif(self.log_video_path / clip_name, fps=fps)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to tensorboard event file
        videos = np.stack(videos[:max_videos_to_save], 0)
        
        # self.log_video(videos, video_title, step, fps=fps) # Disable video logging tensorboard

        # Add the recorded video to the video folder only for evaluation video
        self.log_video_file(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()