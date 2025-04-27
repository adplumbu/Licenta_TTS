import os
import librosa
# La începutul scriptului
import sys
import io
import multiprocessing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from trainer import Trainer, TrainerArgs

from torch.utils.tensorboard import SummaryWriter



from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
# from TTS.tts.configs.shared_configs import CharactersConfig
# from pathlib import Path


# Mută tot codul principal într-o funcție
def main():
    output_path = os.path.dirname(os.path.abspath(__file__))

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_extended.csv",
        path=r"F:\LICENTA2025\BachelorWorkspace\dataset"   
    )
    
    #writer = SummaryWriter(log_dir=os.path.join(output_path, "tensorboard_logs"))

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_romanian_extened_ds_testrun",
        run_description="Running extended data set using ivan, diph1, diph2",
        batch_size=4,
        eval_batch_size=2,
        batch_group_size=2,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=30,
        text_cleaner="basic_cleaners",
        use_phonemes=True,
        phonemizer="espeak",
        phoneme_language="ro",
        phoneme_cache_path=r"F:\LICENTA2025\BachelorWorkspace\dataset\phoneme_cache_extended_ds_ivan_diph1_diph2",
        compute_input_seq_cache=True,
        print_step=30,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=True,
        #characters=character_config,
        save_step=100,
        save_n_checkpoints=3,
        test_sentences=[
            "Un militar român a fost ucis, marţi, în Afganistan, iar un altul a fost rănit.",
            "Vezi două infografii sugestive despre cum sau mişcat plăcile tectonice.",
            "Pentru crearea unei arme nucleare este necesar un uraniu îmbogăţit până la nouăzeci la sută.",
            "Divizată în câteva secţiuni, expoziţia se opreşte asupra vieţii artistului.",
            "Membru al Academiei Regale de Limbă şi Literatură Franceză din Belgia.",
            "Mai precis, a devenit gazda unei emisiuni de divertisment.",
            "Neam amuzat copios în momentul în care am avut confirmarea rezultatului.",
            "Mă refer la erorile în lanţ ale cuplului evocat."
        ]
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # # DEBUG: verificăm ce linie din metadata provoacă eroarea de encoding
    # metadata_path = os.path.join(dataset_config.path, dataset_config.meta_file_train)
    # print(f"Verificăm encoding-ul în fișierul: {metadata_path}")
    # with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
    #     for i, line in enumerate(f):
    #         print(f"{i}: {line.strip()}")

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size = 0.1
        #eval_split_size=config.eval_split_size,
    )

    ###########Verify Dataset###########
    # După încărcarea datelor, afișează statistici
    print(f"Date incarcate: {len(train_samples)} esantioane pentru antrenare, {len(eval_samples)} pentru evaluare.")

    # Verifică durata totală a audio
    total_duration = 0
    for sample in train_samples + eval_samples:
        try:
            audio_path = os.path.join(dataset_config.path, sample["audio_file"])
            y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
            total_duration += len(y) / sr
        except Exception as e:
            print(f"Eroare la procesarea {sample['audio_file']}: {str(e)}")

    print(f"Durata totala a audio: {total_duration/60:.2f} minute")

    # Verifică primele 3 fișiere pentru detalii
    for i, sample in enumerate(train_samples[:3]):
        try:
            audio_path = os.path.join(dataset_config.path, sample["audio_file"])
            y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
            print(f"Fisier {i+1}: {sample['audio_file']}, Durata: {len(y)/sr:.2f}s, Sample rate: {sr}")
        except Exception as e:
            print(f"Eroare la fisierul {sample['audio_file']}: {str(e)}")

    # Verifică primele 3 transcripturi
    for i, sample in enumerate(train_samples[:3]):
        
            print(f"Transcript {i+1}: {sample['text']}")


    ###########Verify Dataset###########

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(gpu=None),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        #dashboard_logger="TensorboardLogger"
    )

    trainer.fit()

# Acest bloc este cheia pentru a rezolva eroarea de multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()