from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

class SpeechToText(ABC):
    @abstractmethod
    def adjust_ambient(self):
        ...

    @abstractmethod
    def transcribe(self) -> str:
        ...

@dataclass(frozen=True)
class Service:
    name: str
    requires_api_key: bool
    online: bool
    model_path: Optional[str] = None
    scorer_path: Optional[str] = None

class Languages(Enum):
    """
    An enumeration of language codes.
    """
    ENGLISH_US = 'en-US'
    ENGLISH_UK = 'en-GB'
    SPANISH_SPAIN = 'es-ES'
    SPANISH_LATIN_AMERICA = 'es-419' #general latin american spanish
    FRENCH_FRANCE = 'fr-FR'
    FRENCH_CANADA = 'fr-CA'
    GERMAN_GERMANY = 'de-DE'
    ITALIAN_ITALY = 'it-IT'
    JAPANESE_JAPAN = 'ja-JP'
    RUSSIAN_RUSSIA = 'ru-RU'
    CHINESE_SIMPLIFIED = 'zh-CN'
    CHINESE_TRADITIONAL = 'zh-TW'
    VIETNAMESE_VIETNAM = 'vi-VN'
    KOREAN_KOREA = 'ko-KR'
    SWEDISH_SWEDEN = 'sv-SE'
    ARABIC_SAUDI_ARABIA = 'ar-SA'
    HINDI_INDIA = 'hi-IN'
    GALICIAN_SPAIN = 'gl-ES'
    CATALAN_SPAIN = 'ca-ES'
    DUTCH_NETHERLANDS = 'nl-NL'
    HEBREW_ISRAEL = 'he-IL'
    POLISH_POLAND = 'pl-PL'
    UKRAINIAN_UKRAINE = 'uk-UA'
    PORTUGUESE_BRAZIL = 'pt-BR'
    PORTUGUESE_PORTUGAL = 'pt-PT'
    BULGARIAN_BULGARIA = 'bg-BG'
    FINNISH_FINLAND = 'fi-FI'
    TURKISH_TURKEY = 'tr-TR'
    CZECH_CZECH_REPUBLIC = 'cs-CZ'
    SLOVAK_SLOVAKIA = 'sk-SK'
    SLOVENIAN_SLOVENIA = 'sl-SI'
    ESTONIAN_ESTONIA = 'et-EE'
    LATVIAN_LATVIA = 'lv-LV'
    ROMANIAN_ROMANIA = 'ro-RO'
    BELARUSIAN_BELARUS = 'be-BY'
    LITHUANIAN_LITHUANIA = 'lt-LT'
    MACEDONIAN_NORTH_MACEDONIA = 'mk-MK'
    GREEK_GREECE = 'el-GR'
    HUNGARIAN_HUNGARY = 'hu-HU'
    SERBIAN_SERBIA = 'sr-RS'
    CROATIAN_CROATIA = 'hr-HR'
    DANISH_DENMARK = 'da-DK'
    NORWEGIAN_BOKMAL_NORWAY = 'nb-NO'
    NORWEGIAN_NYNORSK_NORWAY = 'nn-NO'
    ICELANDIC_ICELAND = 'is-IS'
    MALAY_MALAYSIA = 'ms-MY'
    INDONESIAN_INDONESIA = 'id-ID'
    THAI_THAILAND = 'th-TH'
    SWISS_GERMAN = 'de-CH'
    SWISS_FRENCH = 'fr-CH'
    SWISS_ITALIAN = 'it-CH'
    FILIPINO_PHILIPPINES = 'fil-PH'
    AFRIKAANS_SOUTH_AFRICA = 'af-ZA'
    BASQUE_SPAIN = 'eu-ES'
    IRISH_IRELAND = 'ga-IE'
    WELSH_WALES = 'cy-GB'
    SCOTTISH_GAELIC_SCOTLAND = 'gd-GB'
    MALTEASE_MALTA = 'mt-MT'
    LUKSEMBURGISH_LUXEMBOURG = 'lb-LU'
    YIDDISH = 'yi' #No specific region
    SWAHILI = 'sw' #No specific region
    NEPALI = 'ne-NP'
    BENGALI = 'bn-BD' # or bn-IN
    TAMIL = 'ta-IN'
    TELUGU = 'te-IN'
    MARATHI = 'mr-IN'
    GUJARATI = 'gu-IN'
    PUNJABI = 'pa-IN'
    KANNADA = 'kn-IN'
    MALAYALAM = 'ml-IN'
    ODIA = 'or-IN'
    ASSAMESE = 'as-IN'
    MAITHILI = 'mai-IN'
    SINDHI = 'sd-PK'
    SINHALA = 'si-LK'
    KHMER = 'km-KH'
    LAO = 'lo-LA'
    BURMESE = 'my-MM'
    MONGOLIAN = 'mn-MN'
    HAUSA = 'ha-NG'
    YORUBA = 'yo-NG'
    IGBO = 'ig-NG'
    AMHARIC = 'am-ET'
    SOMALI = 'so-SO'
    RWANDA = 'rw-RW'
    ZULU = 'zu-ZA'
    XHOSA = 'xh-ZA'
    SESOTHO = 'st-ZA'
    TSWANA = 'tn-ZA'
    HAWAIIAN = 'haw-US'
    MAORI = 'mi-NZ'
    SAMOAN = 'sm-WS'
    TONGAN = 'to-TO'
    FIJIAN = 'fj-FJ'
    TAGALOG = 'tl-PH' #old designation, fil-PH is the modern one.

def convert_language_code(language: Languages) -> str:
    """
    Converts a Languages enum value to the two-letter language code expected by Whisper.

    Args:
        language: The Languages enum value.

    Returns:
        The two-letter language code.
    """
    code = language.value
    if '-' in code:
        return code.split('-')[0]
    else:
        return code
