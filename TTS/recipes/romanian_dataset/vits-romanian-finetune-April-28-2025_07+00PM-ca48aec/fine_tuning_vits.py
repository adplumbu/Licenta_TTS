import io
import os
import multiprocessing
import librosa
import sys
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig , CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.downloaders import download_thorsten_de

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.csv", path="F:\\LICENTA2025\\BachelorWorkspace\\dataset\\training\\datasets" 
)


def main():
    
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=True,
        resample=False,
        mel_fmin=0,
        mel_fmax=None 
    )
    character_config=CharactersConfig(
    characters="abcdefghijklmnopqrstuvwxyzăâîșțABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚ -'?!.,",
    punctuations="'?!.,",
    phonemes='iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u02b2\u025a\u02de\u026b\"#$%*+/=abcdefghijklmnopqrstuvwxyzăâîșțABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚ[]^_{}',
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    characters_class="TTS.tts.utils.text.characters.IPAPhonemes",
    )
    config = VitsConfig(
        audio=audio_config,
        run_name="vits-romanian-finetune",
        batch_size=8,
        eval_batch_size=2,
        batch_group_size=4,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=200,
        save_step=1000,
        text_cleaner="basic_cleaners",
        use_phonemes=True,
        phoneme_language="ro",
        characters=character_config,
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        test_sentences=[
            "Un militar rom\u00e2n a fost ucis, mar\u0163i, \u00een Afganistan, iar un altul a fost r\u0103nit.",
            "Vezi dou\u0103 infografii sugestive despre cum sau mi\u015fcat pl\u0103cile tectonice.",
            "Pentru crearea unei arme nucleare este necesar un uraniu \u00eembog\u0103\u0163it p\u00e2n\u0103 la nou\u0103zeci la sut\u0103.",
            "Divizat\u0103 \u00een c\u00e2teva sec\u0163iuni, expozi\u0163ia se opre\u015fte asupra vie\u0163ii artistului.",
            "Membru al Academiei Regale de Limb\u0103 \u015fi Literatur\u0103 Francez\u0103 din Belgia.",
            "Mai precis, a devenit gazda unei emisiuni de divertisment.",
            "Neam amuzat copios \u00een momentul \u00een care am avut confirmarea rezultatului.",
            "M\u0103 refer la erorile \u00een lan\u0163 ale cuplului evocat."
        ],
        output_path=output_path,
        datasets=[dataset_config],
    )

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.05,
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

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
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