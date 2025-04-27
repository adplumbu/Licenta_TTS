import os
import sys
from TTS.utils.synthesizer import Synthesizer

# Setează codificarea corectă pentru consolă și I/O
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Calea către modelul antrenat și configurație
MODEL_PATH = r"F:\LICENTA2025\BachelorWorkspace\modele_antrenate\vits\vits_romanian_small_ds_testrun-April-10-2025_12+13AM-dbf1a08a\best_model_666.pth"
CONFIG_PATH = r"F:\LICENTA2025\BachelorWorkspace\modele_antrenate\vits\vits_romanian_small_ds_testrun-April-10-2025_12+13AM-dbf1a08a\config.json"
OUTPUT_PATH_WAV = r"F:\LICENTA2025\BachelorWorkspace\output_tts\compare_small_vs_extended_ds\small\wav_small_unknown_data"
OUTPUT_PATH_TEST_TRANSCRIPT = r"F:\LICENTA2025\BachelorWorkspace\output_tts\compare_small_vs_extended_ds\small\transcript_small_unknown_data"

# Asigură-te că directorul de ieșire există
os.makedirs(OUTPUT_PATH_WAV, exist_ok=True)
os.makedirs(OUTPUT_PATH_TEST_TRANSCRIPT, exist_ok=True)

# Inițializează sintetizatorul
synthesizer = Synthesizer(
    MODEL_PATH,
    CONFIG_PATH,
    use_cuda=False  # Setează True dacă ai GPU și vrei să accelerezi procesul
)

test_unknown_data = [
    'Totuşi, pare să fie atras de lumina reflectoarelor, adăugând că îi place să fie filmat.', #adr_news_011 - 1
    'Ulterior, a devenit profesor de instrumente tradiţionale, la şcoala populară de artă.', #adr_news_037 - 2
    'O altă regulă este că trebuie să descrii o scenă din natură.', #adr_news_060 - 3
    'Din păcate, trecerea timpului şia pus amprenta asupra vestigiului.', #adr_news_061 - 4
    'Acum noi vorbim doar din auzite, din ce am citit pe internet.', #adr_news_072 - 5
    'Frâna bruscă a dus la răsturnarea maşinii în afara părţii carosabile.', #adr_news_119 - 6
    'Dacă am vrea să facem frumos, near costa zeci de milioane.', #adr_news_132 - 7
    'El a adăugat că nu ştie exact când se vor termina lucrările.', #adr_news_133 - 8
    'El a făcut trimitere la declaraţiile de protest ale liderilor occidentali.', #adr_news_172 - 9
    'Le arăt oamenilor cum era viaţa în trecut, în zona noastră.' #adr_news_210 - 10
]

test_ground_truth = [
    'Pe data de zece octombrie toate depozitele de gaze pe care le are România vor fi pline ochi.', # adr_diph1_024 - 1
    'Tirajul lor este simbolic între două mii şi trei mii de exemplare zilnic.', #adr_diph1_029 - 2
    'Vlad Constantinescu directorul turneului sport arena stritbol.', #adr_diph1_051 - 3
    'Cei care vor săi admire pe câini vor scoate din buzunar cinci lei.', #adr_diph1_122 - 4
    'Zilele viitoare vom reveni cu episodul cârtiţa din clanul interlopilor.', #adr_diph1_231 - 5
    'preţul unui zbor porneşte de la şaptezeci lei', #adr_diph2_104 - 6
    'Nu prea avea timp pentru el însuși.|nu prea avea timp pentru el însuşi.', #adr_diph2_274 - 7
    'băsescu spectacol folcloric la academia de poliţie puneţi mâna pe arme.', #adr_diph2_275 - 8
    'ora doi fără un sfert.', #adr_diph2_276 - 9
    'Dumnezeu a poruncit ca să mănânci trei ani de zile dea rândul numai pădure bătrână. de cea tânără să nu te atingi. înţelesai. Hai, porneşte şiţi fă datoria.', #adr_ivan_181 - 10
    'Mam dus la rai, de la rai la iad, şi de la iad iar la rai.', #adr_ivan_249 - 11
    'în sfârşit, mai stă el Ivan oleacă aşa, cu fruntea rezemată pe mână, şii şi trăsneşte în gând una.', #adr_ivan_253 - 12
    'Taci. Că iam dat de meşteşug.' #adr_ivan_254 - 13
]

# Funcție pentru a normaliza textul românesc
def normalize_romanian_text(text):
    # Asigură-te că toate diacriticele sunt în format corect
    replacements = {
        'ş': 'ș', 'Ş': 'Ș',  # s cu sedilă -> s cu virgulă
        'ţ': 'ț', 'Ţ': 'Ț',  # t cu sedilă -> t cu virgulă
        '\u0219': 'ș', '\u0218': 'Ș',  # alte coduri pentru s cu virgulă
        '\u021B': 'ț', '\u021A': 'Ț',  # alte coduri pentru t cu virgulă
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

# Generează și salvează audio pentru fiecare propoziție
for i, text in enumerate(test_ground_truth):
    # Normalizează textul
    normalized_text = normalize_romanian_text(text)
    
    try:
        print(f"Generez audio pentru: {normalized_text}")
        
        # Generează audio
        outputs = synthesizer.tts(normalized_text)
        
        # Crează numele fișierului de ieșire pentru audio
        file_name = f"test_output_{i+1}.wav"
        file_path = os.path.join(OUTPUT_PATH_WAV, file_name)
        
        # Salvează audio în fișier
        synthesizer.save_wav(outputs, file_path)
        print(f"Audio salvat la: {file_path}")
        
        # Generează și salvează fișierul text corespunzător
        txt_file_name = f"test_output_{i+1}.txt"
        txt_file_path = os.path.join(OUTPUT_PATH_TEST_TRANSCRIPT, txt_file_name)
        
        # Scrie textul în fișier cu codificare UTF-8
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(normalized_text)
        
        print(f"Text salvat la: {txt_file_path}")
    
    except Exception as e:
        print(f"Eroare la procesarea textului: '{normalized_text}'")
        print(f"Detalii eroare: {str(e)}")
        continue  # Continuă cu următoarea propoziție în caz de eroare

print("Procesul de sintetizare a fost finalizat cu succes!")