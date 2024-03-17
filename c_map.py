import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import ipywidgets as widgets
from IPython.display import display, clear_output, Audio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Cursor, RectangleSelector
import matplotlib.colors as mcolors
import logging

from scipy.io import wavfile
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cmap_target = mcolors.LinearSegmentedColormap.from_list("", [(0,0,0,0), (1,0,0,0.5)])


class AudioModel:
    def __init__(self, file_path=None, data=None, sr=None, n_fft=1024, hop_length=None, win_length=None, window='hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        if file_path is not None:
            self.file_path = file_path
            self.sr, self.audio = wavfile.read(file_path)
        elif data is not None and sr is not None:
            self.sr = sr
            self.audio = data
        else:  
            raise ValueError("Either file_path or data and sr must be specified.")
        
        self.dtype = self.audio.dtype
        self.norm = np.max(np.abs(self.audio))
        self.audio_normalized = self.audio.astype(float) / self.norm  # 正規化
        self.f, self.t, self.Sxx = stft(self.audio_normalized, self.sr, nperseg=self.n_fft, noverlap=self.hop_length, nfft=self.win_length, window=self.window)
        

    def time_to_frame_index(self, time):
        return np.abs(self.t - time).argmin()
    
    def time_to_sample_index(self, time):
        idx = int(time * self.sr)
        idx = max(0, idx)
        idx = min(len(self.audio), idx)
        return idx
    
    def get_range(self, audio, start_time, end_time):
        start_index = self.time_to_sample_index(start_time)
        end_index = self.time_to_sample_index(end_time)
        audio_range = audio[start_index:end_index]
        return audio_range
    
    def apply_mask(self, mask):
        return self.Sxx * mask
    
    def apply_mask_audio(self, mask, start_time=None, end_time=None):
        audio = self.reconstruct_audio_with_griffin_lim(self.apply_mask(mask))
        if start_time is not None and end_time is not None:
            audio = self.get_range(audio, start_time, end_time)
        return audio
    
    def apply_mask_mean(self, mask, start_time=None, end_time=None):
        Sxx_masked = self.Sxx * mask
        if start_time is not None and end_time is not None:
            start_index = self.time_to_frame_index(start_time)
            end_index = self.time_to_frame_index(end_time)
            Sxx_masked = Sxx_masked[:, start_index:end_index]
        return np.abs(Sxx_masked).mean(1)

    def apply_inv_mask(self, mask):
        return self.Sxx * (1 - mask)
   
    def apply_inv_mask_audio(self, mask, start_time=None, end_time=None):
        audio = self.reconstruct_audio_with_griffin_lim(self.apply_inv_mask(mask))
        
        if start_time is not None and end_time is not None:
            audio = self.get_range(audio, start_time, end_time)
            
        return audio
    
    def apply_inv_mask_mean(self, mask, start_time=None, end_time=None):
        Sxx_masked_inv = self.Sxx * (1 - mask)
        if start_time is not None and end_time is not None:
            start_index = self.time_to_frame_index(start_time)
            end_index = self.time_to_frame_index(end_time)
            Sxx_masked_inv = Sxx_masked_inv[:, start_index:end_index]
        return np.abs(Sxx_masked_inv).mean(1)
    
    def save_audio(self, audio, output_path=None):
        output_audio = (audio * self.norm).astype(self.audio.dtype)
        if output_path is None:
            # wavのbyte列を取得
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sr, output_audio)
            buffer.seek(0)
            return buffer
        else:
            wavfile.write(output_path, self.sr, output_audio)
        
    # グリフィンリムアルゴリズムを用いた音声復元（初期位相を元の音源の位相として使用）
    def reconstruct_audio_with_griffin_lim(self, Sxx, n_iter=5):
        amplitude = np.abs(Sxx)
        angles = np.exp(1j * np.angle(self.Sxx))
        for _ in range(n_iter):
            _, audio_reconstructed = istft(amplitude * angles, fs=self.sr, nperseg=self.n_fft, noverlap=self.hop_length, nfft=self.win_length, window=self.window)
            _, _, new_Sxx = stft(audio_reconstructed, self.sr, nperseg=self.n_fft, noverlap=self.hop_length, nfft=self.win_length, window=self.window)
            angles = np.exp(1j * np.angle(new_Sxx))
        return audio_reconstructed
    

class AudioViewModel:
    def __init__(self, model):
        self.mask = np.ones(model.Sxx.shape)
        self.model_history = [model]
        self.model_index = 0
        self.reduction_db = 20
        self.target_db = 20
        self.selected_area = [0, 0, 0, 0]  # x1, y1, x2, y2

    @property
    def model(self):
        return self.model_history[self.model_index]
            
    def apply_mask(self):
        return self.model.apply_mask(self.mask)

    def apply_mask_mean(self, start_time=None, end_time=None):
        return self.model.apply_mask_mean(self.mask, start_time, end_time)

    def apply_inv_mask(self):
        return self.model.apply_inv_mask(self.mask)

    def apply_inv_mask_mean(self, start_time=None, end_time=None):
        return self.model.apply_inv_mask_mean(self.mask, start_time, end_time)

    def linear_to_log(self, Sxx):
        return 10 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

    def get_mask_mean_log(self, start_time=None, end_time=None):
        return self.linear_to_log(self.apply_mask_mean(start_time, end_time))
    
    def get_inv_mask_mean_log(self, start_time=None, end_time=None):
        return self.linear_to_log(self.apply_inv_mask_mean(start_time, end_time))
    
    def get_mask_Sxx_log(self):
        return self.linear_to_log(self.apply_mask())

    def get_mask_inv_Sxx_log(self):
        return self.linear_to_log(self.apply_inv_mask())

    def get_audio_from_spectrogram(self, apply_mask=True, start_time=None, end_time=None):
        if apply_mask:
            audio = self.model.apply_mask_audio(self.mask, start_time, end_time)
        else:
            audio = self.model.apply_inv_mask_audio(self.mask, start_time, end_time)
        return audio

    def update_history(self, audio, sr):
        self.model_history = self.model_history[:self.model_index+1]
        model = AudioModel(data=audio, sr=sr)
        self.model_history.append(model)
        if len(self.model_history) > 10:
            self.model_history.pop(0)
        self.model_index = len(self.model_history) - 1
        logger.debug(f"update_history: Sxx_history:{len(self.model_history)}, Sxx_index:{self.model_index}")

    def go_back(self):
        # Sxxの履歴を一つ前に戻す
        if self.model_index > 0:  # 最初の履歴でなければ
            self.model_index -= 1
        self.clear_mask()
        logger.debug("Go back")

    def go_forward(self):
        # Sxxの履歴を一つ進める
        if self.model_index < len(self.model_history) - 1:  # 最新の履歴でなければ
            self.model_index += 1
            self.clear_mask()
        logger.debug("Go forward")

    def select_area(self, x1, y1, x2, y2):
        self.selected_area = [x1, y1, x2, y2]
        logger.debug(f"Selected area: {self.selected_area}")
        
    def clear_area(self):
        self.selected_area = [0, 0, 0, 0]
        logger.debug("Selected area cleared")

    def select_selected(self, b=None):
        # 選択された部分を適用
        self.update_history(self.model.apply_inv_mask_audio(self.mask), self.model.sr)
        # マスクを初期化
        self.clear_mask()
        logger.debug("Selected area applied to selected")

    def select_residual(self, b=None):
        # 残余部分を適用
        self.update_history(self.model.apply_mask_audio(self.mask), self.model.sr)
        # マスクを初期化
        self.clear_mask()
        logger.debug("Selected area applied to residual")

    def get_index_from_time_frq(self):
        start_time, end_time = sorted([self.selected_area[0], self.selected_area[2]])
        start_freq, end_freq = sorted([self.selected_area[1], self.selected_area[3]])
        start_time_index = np.argmin(np.abs(self.model.t - start_time))
        end_time_index = np.argmin(np.abs(self.model.t - end_time))
        start_freq_index = np.argmin(np.abs(self.model.f - start_freq))
        end_freq_index = np.argmin(np.abs(self.model.f - end_freq))
        return start_time_index, end_time_index, start_freq_index, end_freq_index
    
    def create_target_mask(self, invert=False):
        start_time_index, end_time_index, start_freq_index, end_freq_index = self.get_index_from_time_frq()
        
        if invert:
            temp = self.get_mask_inv_Sxx_log()[start_freq_index:end_freq_index, start_time_index:end_time_index]
        else:
            temp = self.get_mask_Sxx_log()[start_freq_index:end_freq_index, start_time_index:end_time_index]
        
        if temp.size == 0:
            return np.zeros_like(self.mask, dtype=bool)
        
        max_power = np.max(temp)
        threshold = max_power - self.target_db
        temp_mask = temp > threshold
        
        target_mask = np.zeros_like(self.mask, dtype=bool)
        target_mask[start_freq_index:end_freq_index, start_time_index:end_time_index][temp_mask] = True

        return target_mask
    
    def apply_noise_reduction(self, invert=False):
        
        if invert:
            reduction_db = -self.reduction_db
        else:
            reduction_db = self.reduction_db

        target_mask = self.create_target_mask(invert=invert)
        
        self.mask[target_mask] *= 10 ** (-reduction_db / 20)
        # マスクの値を0-1に制限
        self.mask[self.mask > 1] = 1
        self.mask[self.mask < 0] = 0
        
        logger.debug(f"Noise reduction applied. inverted {invert}")

    def set_reduction_db(self, reduction_db):
        self.reduction_db = reduction_db
        logger.debug(f"Reduction (dB) changed: {reduction_db}")

    def set_target_db(self, target_db):
        self.target_db = target_db
        logger.debug(f"Target (dB) changed: {target_db}")
        
    def clear_mask(self):
        self.mask = np.ones(self.model.Sxx.shape)
        logger.debug("Mask cleared")


class AudioView:
    def __init__(self, viewModel, interval: int = 500):
        self.viewModel = viewModel
        self.interval = interval
        
        self.ax = []
        self.state_changed = False  # 状態変更フラグ
        self.mouse_released = True  # マウスリリースフラグ
        self.audio_changed = False  # x軸範囲変更フラグ 
        self.setup_fig()
        self.setup_ui()
        self.setup_timer()
        self.start_timer_event = self.fig.canvas.mpl_connect(
            "draw_event", self.first_draw_event)
        self.stop_timer_event = self.fig.canvas.mpl_connect(
            "close_event", self.close_event)
        self.fig.canvas.header_visible = False
        
        self.state_and_audio_change()
        
    def setup_timer(self) -> None:
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.on_timer)  
    
    def close_timer(self) -> None:
        self.timer.remove_callback(self.on_timer)
        self.timer = None
    
    def start_timer(self) -> None:
        self.timer.start()

    def stop_timer(self) -> None:
        self.timer.stop()
    
    def on_timer(self) -> None:
        self.update_spectrogram()
    
    def display(self):
        display(self.output)
        
    def first_draw_event(self, event) -> None:

        self.fig.canvas.mpl_disconnect(self.start_timer_event)
        self.fig.canvas.mpl_connect("draw_event", self.draw_event)
        
        # マウスクリックイベントに対するコールバック関数を設定  
        self.fig.canvas.mpl_connect('button_press_event', self.on_click) 
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.start_timer()
        

    def draw_event(self, event) -> None:
        # xlimの変更を検知
        if self.prev_xlim != self.ax[0].get_xlim():
            self.prev_xlim = self.ax[0].get_xlim()
            self.state_and_audio_change()
        
        self.update_audio_player()
        
    def close_event(self, event) -> None:
        self.close()

    def close(self):
        # タイマーを停止
        if hasattr(self, 'timer') and self.timer:
            self.stop_timer()
            self.close_timer()

        # イベントハンドラを切断
        if hasattr(self, 'fig') and self.fig:
            self.fig.canvas.mpl_disconnect(self.stop_timer_event)
            self.fig.canvas.mpl_disconnect('button_press_event')
            self.fig.canvas.mpl_disconnect('button_release_event')

        # RectangleSelectorがあれば無効化
        if hasattr(self, 'selector') and self.selector:
            self.selector.set_active(False)
        if hasattr(self, 'inv_selector') and self.inv_selector:
            self.inv_selector.set_active(False)

        # matplotlibの図をクローズ
        if hasattr(self, 'fig') and self.fig:
            plt.close(self.fig)

        # ウィジェットのクリーンアップ
        if hasattr(self, 'audio_output') and self.audio_output:
            self.audio_output.clear_output()
            
        if hasattr(self, 'output') and self.output:
            self.output.clear_output()
        
            
    def setup_fig(self):
        
        # グリッドの設定
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 0.1])  # カラーマップの横幅をスペクトルの平均値の横幅の3倍に設定
        gs.update(wspace=0.2)
        
        self.fig = plt.figure(figsize=(10, 5))

        # 最初のサブプロットを作成
        self.ax.append(self.fig.add_subplot(gs[0]))
        # 2番目のサブプロットを作成し、x軸をax1と連動
        self.ax.append(self.fig.add_subplot(gs[3], sharex=self.ax[0], sharey=self.ax[0]))
        # 3番目のサブプロットを作成し、y軸をax1と連動
        self.ax.append(self.fig.add_subplot(gs[1], sharey=self.ax[0]))
        # 4番目のサブプロットを作成し、x軸とy軸をそれぞれax2とax3と連動
        self.ax.append(self.fig.add_subplot(gs[4], sharex=self.ax[2], sharey=self.ax[0]))
        
        # サブプロット間の隙間を調整
        self.fig.subplots_adjust(wspace=0.0001)

        settings = {
            'xlim': [0, self.viewModel.model.t[-1]],  # x軸の範囲を指定
            'ylabel': 'Frequency [kHz]'
        }
        cmap_settings = {
            'aspect': 'auto',
            'origin': 'lower',
            'interpolation': 'none',
            'rasterized': True,
            'extent': [self.viewModel.model.t.min(), self.viewModel.model.t.max(), self.viewModel.model.f.min()/1e3, self.viewModel.model.f.max()/1e3]
        }
        self.cmap, self.cmap_mask=[], []
        t, f = self.viewModel.model.t, self.viewModel.model.f/1E3
        Sxx_log = self.viewModel.get_mask_Sxx_log()
        self.cmap.append(self.ax[0].imshow(Sxx_log, **cmap_settings))
        self.cmap_mask.append(self.ax[0].imshow(np.zeros(Sxx_log.shape), vmin=0, vmax=1, cmap=cmap_target, **cmap_settings))
        inv_Sxx_log = self.viewModel.get_mask_inv_Sxx_log()
        vmin, vmax = self.cmap[0].get_clim()
        self.cmap.append(self.ax[1].imshow(inv_Sxx_log, vmin=vmin, vmax=vmax, **cmap_settings))
        self.cmap_mask.append(self.ax[1].imshow(np.zeros(inv_Sxx_log.shape), vmin=0, vmax=1, cmap=cmap_target, **cmap_settings))
        
        self.ax[0].set(**settings)
        self.ax[1].set(xlabel="Time [s]" , **settings)
        plt.colorbar(self.cmap[0], ax=self.ax[2], label='Power[dB]')
        plt.colorbar(self.cmap[1], ax=self.ax[3], label='Power[dB]')
        self.prev_xlim = self.ax[0].get_xlim()
        self.prev_ylim = self.ax[0].get_ylim()
        
        # RectangleSelectorの設定
        props = dict(
            facecolor=None, 
            alpha=0.5)
        self.selector = RectangleSelector(self.ax[0], self.onselect, useblit=True,
                                        button=[1],  # 左クリックのみ有効
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        props=props,
                                        interactive=True)
        
        self.inv_selector = RectangleSelector(self.ax[1], self.onselect, useblit=True,
                                        button=[1],  # 左クリックのみ有効
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        props=props,
                                        interactive=True)
        
        # 平均値をプロット（初期化）
        self.mean_line1, = self.ax[2].plot([], [], label='Spectrogram (Removed)')
        self.mean_line2, = self.ax[3].plot([], [], label='Spectrogram (Inverted)')
        self.ax[2].grid(True)
        self.ax[2].set( ylim=self.prev_ylim)
        self.ax[3].grid(True)
        self.ax[3].set(ylim=self.prev_ylim)
        
        # 十字カーソルを表示するための設定
        # self.cursor = []
        # for ax in self.ax:
        #     self.cursor.append(Cursor(ax, useblit=True, color='red', linewidth=1))
    
    def setup_ui(self):
    
        self.output = widgets.Output()
        self.audio_output = widgets.Output()
        self.target_db_slider = widgets.IntSlider(min=0, max=60, step=1, value=20, continuous_update=False, description='Target (dB):')
        self.target_db_slider.observe(self.on_target_db_slider_value_changed, names='value')
        self.reduction_db_slider = widgets.IntSlider(min=0, max=60, step=1, value=20, continuous_update=False, description='Reduction (dB):')
        self.reduction_db_slider.observe(self.on_reduction_db_slider_value_changed, names='value')
        self.apply_button = widgets.Button(description="Apply Reduction")
        self.apply_button.on_click(self.apply_noise_reduction)
        self.save_button = widgets.Button(description="Save Audio")
        self.save_button.on_click(self.save_audio)
        self.select_residual_button = widgets.Button(description="Select Residual")
        self.select_residual_button.on_click(self.select_residual)
        self.select_selected_button = widgets.Button(description="Select selected")
        self.select_selected_button.on_click(self.select_selected)
        self.go_back_button = widgets.Button(description="Go Back")
        self.go_back_button.on_click(self.go_back)
        self.go_forward_button = widgets.Button(description="Go Forward")
        self.go_forward_button.on_click(self.go_forward)
        self.clear_mask_button = widgets.Button(description="Clear Mask")
        self.clear_mask_button.on_click(self.clear_mask)

        vb_reduction = widgets.HBox([self.target_db_slider, self.reduction_db_slider, self.apply_button])
        vb_select = widgets.HBox([self.select_residual_button, self.select_selected_button, self.clear_mask_button])
        vb_bf = widgets.HBox([self.go_back_button, self.go_forward_button])

        with self.output:
            display(widgets.VBox([self.audio_output, vb_bf, vb_select, vb_reduction]))
            plt.show()
        
        
    
    def state_and_audio_change(self):
        self.state_changed = True
        self.audio_changed = True
        
    def onselect(self, eclick, erelease):

        logger.debug(f"onselect: {eclick.inaxes}")
        if self.ax[0] == eclick.inaxes or self.ax[1] == eclick.inaxes:
            # 選択された領域の始点と終点を記録
            self.viewModel.select_area(eclick.xdata,
                                       eclick.ydata * 1000,
                                       erelease.xdata,
                                       erelease.ydata * 1000)
            self.state_changed = True
            logger.debug(f"Selected area: {self.viewModel.selected_area}")
 
    def on_click(self, event):
        self.mouse_released = False
        # グラフがクリックされたときにselectorをアクティブにする
        if event.inaxes == self.ax[0]:
            self.inv_selector.clear()
            self.inv_selector.set_active(False)
            if not self.selector.active:
                self.viewModel.clear_area()
                self.selector.set_active(True)
            
            
        elif event.inaxes == self.ax[1]:
            self.selector.clear()
            self.selector.set_active(False)
            if not self.inv_selector.active:
                self.viewModel.clear_area()
                self.inv_selector.set_active(True)
                    
    def on_release(self, event):
        self.mouse_released = True
            
    def clear_mask(self, b=None):
        self.viewModel.clear_mask()
        self.state_and_audio_change()
        logger.debug("Mask cleared")

    def go_back(self, b=None):
        self.viewModel.go_back()
        self.state_and_audio_change()
        logger.debug("Go back")

    def go_forward(self, b=None):
        self.viewModel.go_forward()
        self.state_and_audio_change()
        logger.debug("Go forward")

    def select_selected(self, b=None):
        self.viewModel.select_selected()
        self.state_and_audio_change()
        logger.debug("Selected area applied to selected")

    def select_residual(self, b=None):
        self.viewModel.select_residual()
        self.state_and_audio_change()
        logger.debug("Selected area applied to residual")

    def on_target_db_slider_value_changed(self, change):
        self.viewModel.set_target_db(change['new'])
        self.state_changed = True
        logger.debug(f"Target (dB) changed: {change['new']}")
                    
    def on_reduction_db_slider_value_changed(self, change):
        self.viewModel.set_reduction_db(change['new'])
        self.state_changed = True
        logger.debug(f"Reduction (dB) changed: {change['new']}")

    def apply_noise_reduction(self, b):
        if self.selector.visible:
            self.viewModel.apply_noise_reduction()
        else: # 残余部分に適用
            self.viewModel.apply_noise_reduction(invert=True)
        self.state_and_audio_change()
        logger.debug("Noise reduction applied")

    def save_audio(self, b=None):
        # output_path = "modified_audio.wav"
        audio = self.viewModel.get_audio_from_spectrogram()
        byte_audio = self.viewModel.model.save_audio(audio)
        
        # print(f"Audio saved to {output_path}")
        # logger.debug(f"Audio saved to {output_path}")
        
        return byte_audio

    def update_spectrogram(self):
        if not self.state_changed or not self.mouse_released:
            return
        self.state_changed = False
        
        # maskを適用
        Sxx_log = self.viewModel.get_mask_Sxx_log()  # マスクを適用したスペクトログラム   
        Sxx_log_removed = self.viewModel.get_mask_inv_Sxx_log()  # 除去された音源

        # 現在の時間範囲を取得
        start, end = self.ax[0].get_xlim()
    
        # 平均値を計算
        mean_Sxx_log = self.viewModel.get_mask_mean_log(start, end)
        mean_Sxx_log_inv = self.viewModel.get_inv_mask_mean_log(start, end)

        # カラーマップの更新して再描画
        self.cmap[0].set_data(Sxx_log)
        self.cmap[1].set_data(Sxx_log_removed)
        
        # マスクのプレビューを更新
        self.cmap_mask[0].set_data(np.zeros(Sxx_log.shape))
        self.cmap_mask[1].set_data(np.zeros(Sxx_log_removed.shape))
        if self.selector.visible:
            prev_mask = self.viewModel.create_target_mask()
            self.cmap_mask[0].set_data(prev_mask)
        elif self.inv_selector.visible:
            prev_mask = self.viewModel.create_target_mask(invert=True)
            self.cmap_mask[1].set_data(prev_mask)
        
        self.mean_line1.set_xdata(mean_Sxx_log)
        self.mean_line1.set_ydata(self.viewModel.model.f/1E3)
        self.mean_line2.set_xdata(mean_Sxx_log_inv)
        self.mean_line2.set_ydata(self.viewModel.model.f/1E3)
        # x軸をオートスケール
        self.ax[2].relim()
        self.ax[2].autoscale_view(scalex=True, scaley=False)
        self.ax[3].set_xlim(*self.ax[2].get_xlim())

        self.fig.canvas.draw_idle()
        logger.debug("Spectrogram updated")
        

    def update_audio_player(self):
        if not self.audio_changed or not self.mouse_released:
            return
        self.audio_changed = False
        
        start, end = self.ax[0].get_xlim()  # スペクトログラムの表示範囲を取得      
        
        audio = self.viewModel.get_audio_from_spectrogram(start_time=start, end_time=end)
        inv_audio = self.viewModel.get_audio_from_spectrogram(False, start_time=start, end_time=end)
        sr = self.viewModel.model.sr
             
        audio_player = Audio(audio, rate=sr)
        inv_audio_player = Audio(inv_audio, rate=sr)
                    
        with self.audio_output:
            clear_output(wait=True)
            display(audio_player, inv_audio_player)
        logger.debug("Audio player updated")
        
        # self.start_timer()


class InteractiveSoundEditor:
    def __init__(self, initial_file_path=None):
        self.audio_view = None  # AudioViewのインスタンスを保持する変数
        self.setup_ui()
        if initial_file_path is not None:
            self.load_and_display_audio(initial_file_path)

    def setup_ui(self):
        self.file_upload = widgets.FileUpload(accept='.wav', multiple=False)
        self.file_upload.observe(self.on_file_upload, names='value')
        self.output_area = widgets.Output()  # 出力エリアを設定
    
    def on_file_upload(self, change):
        # アップロードされたファイルの処理
        uploaded_file = next(iter(self.file_upload.value))
        uploaded_filename = uploaded_file['name']
        content = uploaded_file['content']
        with open(uploaded_filename, 'wb') as f:
            f.write(content)
        
       # 新しいAudioModel, AudioViewModel, そしてAudioViewインスタンスを作成して表示
        self.load_and_display_audio(uploaded_filename)

    def load_and_display_audio(self, file_path):
        # 既存の表示をクリア
        if self.audio_view is not None:
            # 必要ならば、AudioViewクラス内でリソースのクリーンアップを行うメソッドを呼び出す
            self.audio_view.close()
        # 新しいAudioModel, AudioViewModel, そしてAudioViewインスタンスを作成
        model = AudioModel(file_path)
        view_model = AudioViewModel(model)
        self.audio_view = AudioView(view_model)
        
        # UIを再描画
        with self.output_area:
            clear_output(wait=True)  # 既存の出力をクリア
            display(widgets.VBox([self.file_upload,self.audio_view.output]))
        
    def display(self):
        display(self.output_area)
            
            
if __name__ == "__main__":
    import panel as pn
    file_path = 'sample/Example_4_mix.wav'  # WAVファイルのパスを指定
    output_path = 'sample/output_audio.wav'  # 出力ファイルのパスを指定
    
    ise = InteractiveSoundEditor(file_path)
    ise.display()
    pn.panel(ise.output_area).show()