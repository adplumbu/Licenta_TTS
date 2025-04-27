import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import wave
import os
from datetime import datetime
import time

def load_waveform(filename):
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio = wf.readframes(n_frames)
        waveform = np.frombuffer(audio, dtype=np.int16)
        time_axis = np.linspace(0, n_frames / sample_rate, num=n_frames)
    return time_axis, waveform, sample_rate

def trim_silence_for_plot(waveform, time_axis, threshold=500, margin=100):
    abs_wave = np.abs(waveform)
    above_thresh = np.where(abs_wave > threshold)[0]
    if len(above_thresh) == 0:
        return time_axis, waveform
    start = max(0, above_thresh[0] - margin)
    return time_axis[start:], waveform[start:]

# Căile către fișierele audio
# Notă: Înlocuiește aceste căi cu locațiile corecte ale fișierelor tale
ground_truth_file = r'F:\LICENTA2025\BachelorWorkspace\dataset\training\wavs\adr_diph2_104.wav'
small_model_file = r'F:\LICENTA2025\BachelorWorkspace\output_tts\compare_small_vs_extended_ds\small\01_w_wav_small_ground_truth\test_output_6.wav'
extended_model_file = r'F:\LICENTA2025\BachelorWorkspace\output_tts\compare_small_vs_extended_ds\extended\01_w_wav_extended_ground_truth\test_output_6.wav'

# Încarcă semnalele audio
time_gt, waveform_gt, sr_gt = load_waveform(ground_truth_file)
time_small, waveform_small, sr_small = load_waveform(small_model_file)
time_ext, waveform_ext, sr_ext = load_waveform(extended_model_file)

# Valori inițiale pentru threshold și margin
initial_threshold = 500
initial_margin = 100
max_display_time = 10.0  # Maxim 10 secunde de afișare

# Configurarea plot-ului
fig, axs = plt.subplots(4, 1, figsize=(15, 12))
plt.subplots_adjust(bottom=0.25)

# Funcția de actualizare pentru slider-e
def update(val):
    threshold = threshold_slider.val
    margin = int(margin_slider.val)

    # Aplicăm trim_silence pentru ground truth
    time_trimmed, wave_trimmed = trim_silence_for_plot(waveform_gt, time_gt, threshold, margin)
    
    # Determinăm lungimea minimă pentru comparație
    min_len = min(len(wave_trimmed), len(waveform_small), len(waveform_ext))
    
    t_trimmed = time_trimmed[:min_len]
    wave_trimmed = wave_trimmed[:min_len]
    small_trimmed = waveform_small[:min_len]
    ext_trimmed = waveform_ext[:min_len]
    
    # Limitarea la max_display_time secunde
    if len(t_trimmed) > 0:
        max_time_idx = min(len(t_trimmed), np.where(t_trimmed > max_display_time)[0][0] if np.any(t_trimmed > max_display_time) else len(t_trimmed))
        t_view = t_trimmed[:max_time_idx]
        wave_view = wave_trimmed[:max_time_idx]
        small_view = small_trimmed[:max_time_idx]
        ext_view = ext_trimmed[:max_time_idx]
        
        # Actualizăm informațiile despre durata totală a semnalului
        if len(t_trimmed) > 0:
            signal_info.set_text(f'Durată semnal: {t_trimmed[-1]:.2f}s (afișare limitată la {max_display_time:.1f}s)')
        
        # Actualizăm plot-urile
        axs[0].cla()
        axs[0].plot(t_view, wave_view, color='blue')
        axs[0].set_title('Ground Truth (trimmed)')
        axs[0].set_xlim(0, max_display_time)
        
        axs[1].cla()
        axs[1].plot(t_view, small_view, color='green')
        axs[1].set_title('Model antrenat pe set mic (10 epoci)')
        axs[1].set_xlim(0, max_display_time)
        
        axs[2].cla()
        axs[2].plot(t_view, ext_view, color='red')
        axs[2].set_title('Model antrenat pe set extins (30 epoci)')
        axs[2].set_xlim(0, max_display_time)
        
        axs[3].cla()
        axs[3].plot(t_view, wave_view, label='Ground Truth', color='blue', alpha=0.5)
        axs[3].plot(t_view, small_view, label='Model set mic', color='green', alpha=0.5)
        axs[3].plot(t_view, ext_view, label='Model set extins', color='red', alpha=0.5)
        axs[3].legend()
        axs[3].set_title('Comparare suprapusă a celor trei semnale')
        axs[3].set_xlim(0, max_display_time)
        
        for ax in axs:
            ax.set_xlabel('Timp (secunde)')
            ax.set_ylabel('Amplitudine')

    fig.canvas.draw_idle()

# Sliders
ax_threshold = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_margin = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_save = plt.axes([0.35, 0.02, 0.3, 0.05])  # Poziția butonului de salvare

threshold_slider = Slider(ax_threshold, 'Threshold', 0, 5000, valinit=initial_threshold, valstep=100)
margin_slider = Slider(ax_margin, 'Margin', 0, 1000, valinit=initial_margin, valstep=10)
save_button = Button(ax_save, 'Salvează Imaginea')

# Adăugăm un text pentru a afișa informații despre durata semnalului
ax_info = plt.figtext(0.5, 0.2, '', ha='center')
signal_info = ax_info

# Path-ul pentru salvarea imaginii
save_path = r'F:\LICENTA2025\BachelorWorkspace\output_tts\compare_small_vs_extended_ds\plots'

# Funcția pentru salvarea imaginii
def save_figure(event):
    # Creează directorul dacă nu există
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extrage numele fișierului audio din calea completă
    # Folosim extended_model_file ca bază pentru denumire
    base_filename = os.path.basename(extended_model_file)
    wav_name = os.path.splitext(base_filename)[0]  # Elimină extensia .wav
    
    # Adaugă un timestamp pentru a evita suprascrierea
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f'compare_{wav_name}_{timestamp}.png'
    filepath = os.path.join(save_path, filename)
    
    # Ajustează figura pentru salvare (fără slider-e și butoane)
    plt.figure(figsize=(15, 12))
    
    # Recreează graficele
    new_fig, new_axs = plt.subplots(4, 1, figsize=(15, 12))
    
    # Obține valorile curente
    threshold = threshold_slider.val
    margin = int(margin_slider.val)
    
    # Aplică trim_silence pentru ground truth
    time_trimmed, wave_trimmed = trim_silence_for_plot(waveform_gt, time_gt, threshold, margin)
    
    # Determină lungimea minimă pentru comparație
    min_len = min(len(wave_trimmed), len(waveform_small), len(waveform_ext))
    
    t_trimmed = time_trimmed[:min_len]
    wave_trimmed = wave_trimmed[:min_len]
    small_trimmed = waveform_small[:min_len]
    ext_trimmed = waveform_ext[:min_len]
    
    # Limitarea la max_display_time secunde
    max_time_idx = min(len(t_trimmed), np.where(t_trimmed > max_display_time)[0][0] if np.any(t_trimmed > max_display_time) else len(t_trimmed))
    t_view = t_trimmed[:max_time_idx]
    wave_view = wave_trimmed[:max_time_idx]
    small_view = small_trimmed[:max_time_idx]
    ext_view = ext_trimmed[:max_time_idx]
    
    # Extrage numele fișierelor pentru titluri
    ground_truth_name = os.path.basename(ground_truth_file)
    small_model_name = os.path.basename(small_model_file)
    extended_model_name = os.path.basename(extended_model_file)
    
    # Creează graficele pentru salvare
    new_axs[0].plot(t_view, wave_view, color='blue')
    new_axs[0].set_title(f'Ground Truth: {ground_truth_name}')
    new_axs[0].set_xlim(0, max_display_time)
    
    new_axs[1].plot(t_view, small_view, color='green')
    new_axs[1].set_title(f'Model set mic (10 epoci): {small_model_name}')
    new_axs[1].set_xlim(0, max_display_time)
    
    new_axs[2].plot(t_view, ext_view, color='red')
    new_axs[2].set_title(f'Model set extins (30 epoci): {extended_model_name}')
    new_axs[2].set_xlim(0, max_display_time)
    
    new_axs[3].plot(t_view, wave_view, label='Ground Truth', color='blue', alpha=0.5)
    new_axs[3].plot(t_view, small_view, label='Model set mic', color='green', alpha=0.5)
    new_axs[3].plot(t_view, ext_view, label='Model set extins', color='red', alpha=0.5)
    new_axs[3].legend()
    new_axs[3].set_title('Comparare suprapusă a celor trei semnale')
    new_axs[3].set_xlim(0, max_display_time)
    
    for ax in new_axs:
        ax.set_xlabel('Timp (secunde)')
        ax.set_ylabel('Amplitudine')
    
    # Adaugă informații despre threshold și margin în titlul figurii
    durata_reala = t_trimmed[-1] if len(t_trimmed) > 0 else 0
    plt.suptitle(f'Comparare semnale audio (Threshold: {threshold}, Margin: {margin}, Durată reală: {durata_reala:.2f}s, Afișare: 0-{max_display_time:.1f}s)', fontsize=12)
    
    plt.tight_layout()
    
    # Salvează figura
    new_fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(new_fig)
    
    print(f"Imaginea a fost salvată la: {filepath}")

# Inițializare
update(None)
threshold_slider.on_changed(update)
margin_slider.on_changed(update)
save_button.on_clicked(save_figure)

plt.tight_layout(rect=[0, 0.25, 1, 1])  # Ajustare layout pentru a face loc slider-elor și butonului
plt.show()