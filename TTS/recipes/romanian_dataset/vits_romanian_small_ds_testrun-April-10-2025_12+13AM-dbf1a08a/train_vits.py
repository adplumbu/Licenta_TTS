import os
import librosa
# La începutul scriptului
import sys
import io
import multiprocessing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from trainer import Trainer, TrainerArgs

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
        meta_file_train="metadata.csv",
        path=r"F:\LICENTA2025\BachelorWorkspace\dataset"   
    )

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
        run_name="vits_romanian_small_ds_testrun",
        run_description="Desc",
        batch_size=4,
        eval_batch_size=2,
        batch_group_size=2,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=10,
        text_cleaner="basic_cleaners",
        use_phonemes=True,
        phonemizer="espeak",
        phoneme_language="ro",
        phoneme_cache_path=r"F:\LICENTA2025\BachelorWorkspace\dataset\phoneme_cache_new_try",
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        #characters=character_config,
        save_step=50,
        save_n_checkpoints=3,
        test_sentences=[
            "Astazi am plecat la facultate.",
            "Am plecat la cursul de Introducere in Automatica."
        ]
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
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
    )

    trainer.fit()

# Acest bloc este cheia pentru a rezolva eroarea de multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()