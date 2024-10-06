from pyscript import display
import ipywidgets as widgets
import js
import panel as pn
import c_map
from IPython.display import display, clear_output
import matplotlib
import matplotlib.style as mplstyle
import asyncio
import logging
import numpy as np

mplstyle.use("fast")
# js.pyodide.setDebug(True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# pn.extension(sizing_mode="stretch_width")

print("cmap_main.py loaded")


def set_create_root_element(root_element, fig: matplotlib.figure) -> matplotlib.figure:
    div = js.document.createElement("div")
    root_element.appendChild(div)

    def cre(self):
        return div

    fig.canvas.create_root_element = cre.__get__(  # type: ignore
        cre, fig.canvas.__class__
    )

    return fig


class AudioViewPs(c_map.AudioView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_timer(self) -> None:
        self.timer = None

    def close_timer(self) -> None:
        pass

    def start_timer(self) -> None:
        pass

    def stop_timer(self) -> None:
        pass

    def setup_ui(self):
        self.fig_pane = pn.pane.HTML(f"""
            <div id="cmap(appendChild)"></div>
        """)

        self.org_audio_player = pn.pane.Audio(
            np.zeros(0), sample_rate=self.viewModel.model.sr
        )
        self.audio_player = pn.pane.Audio(
            np.zeros(0), sample_rate=self.viewModel.model.sr
        )
        self.inv_audio_player = pn.pane.Audio(
            np.zeros(0), sample_rate=self.viewModel.model.sr
        )
        self.audio_player_pane = pn.Row(
            self.org_audio_player, self.audio_player, self.inv_audio_player
        )

        self.target_db_slider = pn.widgets.FloatSlider(
            name="Target dB", start=0, end=60, step=1, value=20
        )
        self.target_db_slider.param.watch(
            self.on_target_db_slider_value_changed, "value_throttled"
        )
        self.reduction_db_slider = pn.widgets.FloatSlider(
            name="Reduction dB", start=0, end=60, step=1, value=20
        )
        self.reduction_db_slider.param.watch(
            self.on_reduction_db_slider_value_changed, "value_throttled"
        )
        self.apply_button = pn.widgets.Button(name="Apply Reduction")
        self.apply_button.on_click(self.apply_noise_reduction)
        self.save_button = pn.widgets.FileDownload(
            callback=pn.bind(self.save_audio), filename="output.wav"
        )
        self.select_residual_button = pn.widgets.Button(name="Select Residual")
        self.select_residual_button.on_click(self.select_residual)
        self.select_selected_button = pn.widgets.Button(name="Select Selected")
        self.select_selected_button.on_click(self.select_selected)
        self.go_back_button = pn.widgets.Button(name="Go Back")
        self.go_back_button.on_click(self.go_back)
        self.go_forward_button = pn.widgets.Button(name="Go Forward")
        self.go_forward_button.on_click(self.go_forward)
        self.clear_mask_button = pn.widgets.Button(name="Clear Mask")
        self.clear_mask_button.on_click(self.clear_mask)

        self.enhance_percussive_button = pn.widgets.Button(name="Enhance Percussive")
        self.enhance_percussive_button.on_click(self.enhance_percussive)
        self.enhance_harmonic_button = pn.widgets.Button(name="Enhance Harmonic")
        self.enhance_harmonic_button.on_click(self.enhance_harmonic)
        self.enhance_anomaly_button = pn.widgets.Button(name="Enhance Anomaly")
        self.enhance_anomaly_button.on_click(self.enhance_anomaly)
        self.enhance_anomaly_threshold = pn.widgets.FloatSlider(
            start=0, end=10, step=0.5, value=3
        )

        self.vb_reduction = pn.Row(
            self.target_db_slider,
            self.reduction_db_slider,
            self.apply_button,
        )

        self.vb_select = pn.Row(
            self.select_residual_button,
            self.select_selected_button,
            self.clear_mask_button,
        )

        self.vb_bf = pn.Row(
            self.go_back_button, self.go_forward_button, self.save_button
        )

        self.vb_enhance = pn.Row(
            self.enhance_percussive_button,
            self.enhance_harmonic_button,
        )
        self.vb_enhance_anomaly = pn.Row(
            self.enhance_anomaly_button, self.enhance_anomaly_threshold
        )

        self.output_area = pn.Column(
            self.audio_player_pane,
            self.vb_bf,
            self.vb_select,
            self.vb_reduction,
            self.vb_enhance,
            self.vb_enhance_anomaly,
            self.fig_pane,
        )

    def on_target_db_slider_value_changed(self, event):
        super().on_target_db_slider_value_changed(dict(new=event.new))

    def on_reduction_db_slider_value_changed(self, event):
        super().on_reduction_db_slider_value_changed(dict(new=event.new))

    def update_spectrogram(self):
        return super().update_spectrogram()

    def update_audio_player(self):
        if not self.audio_changed or not self.mouse_released:
            return
        self.audio_changed = False

        start, end = self.ax[0].get_xlim()  # スペクトログラムの表示範囲を取得

        org_audio = self.viewModel.model.get_range(
            self.viewModel.model.audio, start, end
        )
        audio = self.viewModel.get_audio_from_spectrogram(
            start_time=start, end_time=end
        )
        inv_audio = self.viewModel.get_audio_from_spectrogram(
            False, start_time=start, end_time=end
        )

        self.org_audio_player.object = org_audio
        self.audio_player.object = audio
        self.inv_audio_player.object = inv_audio

        logger.debug("Audio player updated")

    def close(self):
        self.output_area.clear()
        element = js.document.getElementById(self.fig.canvas._id)
        element.parentNode.removeChild(element)

        return super().close()

    def display(self, target="app"):
        self.output_area.servable(target=target)

    def display_fig(self, target="fig"):
        set_create_root_element(js.document.getElementById(target), self.fig)
        plt.show()


class InteractiveSoundEditorPs(c_map.InteractiveSoundEditor):
    def setup_ui(self, app_target="app", fig_target="fig", target="file_upload"):
        self.app_target = app_target
        self.fig_target = fig_target
        self.target = target
        self.file_upload = pn.widgets.FileInput(accept=".wav", multiple=False)
        self.file_upload.param.watch(self.on_file_upload, "value")
        self.view = pn.Column(self.file_upload)

    def on_file_upload(self, change):
        # アップロードされたファイルの処理

        if self.file_upload.value is not None:
            self.file_upload.save("temp.wav")

            # 新しいAudioModel, AudioViewModel, そしてAudioViewインスタンスを作成して表示
            self.load_and_display_audio("temp.wav")

    def load_and_display_audio(self, file_path):
        # 既存の表示をクリア
        if self.audio_view is not None:
            # 必要ならば、AudioViewクラス内でリソースのクリーンアップを行うメソッドを呼び出す
            self.audio_view.close()
        # 新しいAudioModel, AudioViewModel, そしてAudioViewインスタンスを作成
        model = c_map.AudioModel(file_path)
        view_model = c_map.AudioViewModel(model)
        self.audio_view = AudioViewPs(view_model)

        # UIを再描画
        self.audio_view.display("app")
        self.audio_view.display_fig("fig")

    def update(self):
        self.audio_view.update_spectrogram()

    def close(self):
        self.view.clear()
        self.audio_view.close()

    def display(self):
        self.view.servable(target=self.target)


import matplotlib.pyplot as plt

file_path = "./Example_4_mix.wav"

se = InteractiveSoundEditorPs(file_path)
se.display()

function_list = [se.update]


async def foo():
    while True:
        await asyncio.sleep(0.5)
        [f() for f in function_list]


await foo()

# se = InteractiveSoundEditorPs(file_path)
# se.display()
# display(se.output_area, target="output")
