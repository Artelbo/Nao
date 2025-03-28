from google.genai import types
from typing import List, Dict, Any
from stt import STT, SimpleService, Services
from stt.base import Languages
import speech_recognition as sr
from thefuzz import fuzz
import os
from google import genai
import textwrap
import time
import locale

locale.setlocale(locale.LC_TIME, Languages.ITALIAN_ITALY.value.replace('-', '_'))

start = 'leonardo'
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
system_prompt = '''
Sei Nao, un robot che risponde grazie a Google Gemini, sei stato programmato grazie a Python e la mente di 5 geni.
Le risposte che fornisci devono essere in testo semplice, senza alcuna formattazione stilistica. Evita l'uso di grassetto, corsivo, sottolineato, elenchi puntati o numerati, parentesi, o qualsiasi altro tipo di formattazione.
Fornisci risposte chiare e concise, esclusivamente in testo puro.
Le risposte devono essere brevi e riassuntive.
Ricorda di riferiti a me (io) come tu, sempre, eccetto quando puoi sott'intenderlo.
Se ti viene posta una domanda non pertinente puoi rispondere con: "Non ho capito, potresti ripetere?"
Ora locale: {ora}
Data locale: {data}

Conversazioni Precedenti:
---
{memoria}
---
'''

class Memory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.memory = []

    def add(self, q, a) -> None:
        self.memory.append({'question': q, 'answer': a})
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def get_memory_string(self) -> str:
        return '\n\n'.join([f'Domanda: {item['question']}\nRisposta: {item['answer']}' for item in self.memory])

memory = Memory()

def answer(r: str) -> None:
    print(f' -> {r}')

    memory_string = memory.get_memory_string()

    # Include memory in the system prompt
    modified_system_prompt = system_prompt.format(
        ora=time.strftime('%H:%M:%S'),
        data=time.strftime('%A, %d %B %Y'),
        memoria=memory_string
    )


    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=r,
        config=types.GenerateContentConfig(
            system_instruction=modified_system_prompt
        )
    )

    response_text = response.text
    print('\n'.join([f' <- {x}' for x in textwrap.wrap(response_text, width=50, break_long_words=False)]),
          end='\n\n')

    memory.add(r, response_text)

with sr.Microphone() as source:
    print('Loading, please wait...')
    stt = STT(source, SimpleService.BestOnline)
    print('Adjusting for ambient noise')
    stt.adjust_ambient()
    print('Ready to start speaking...')
    while True:
        for text in stt.transcribe(None):
            t = text.lower().replace('ehi', 'hey', 1)
            # print(f'[] {t}')
            if not (fuzz.ratio(t[0:len(start)], start.lower()) > 50):
                continue
            t = t[len(start)+1:].strip()
            if t.startswith(tuple(',.;:')):
                t = t[1:]
                t = t.strip()
            if len(t) <= 0:
                break
            answer(t)

        print('Si?')
        for text in stt.transcribe(None):
            t = text.lower()
            if len(t.strip()) <= 0:
                continue

            answer(t)
            break