# settings.py
import os
import logging.config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERBOSE = False

PRECISO_EXPLICAR = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
        "not_too_simple": {
            "format": "{levelname} {name} {funcName}: {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": os.path.join(BASE_DIR, "debug.log"),
            "formatter": "verbose",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "not_too_simple",
        },
    },
    "loggers": {
        "top_module": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False, # Se true propaga para o root (parent) pode duplicar
        },
        "top_module.child": {   # Logger filho de top_module
            "level": "DEBUG",    # <- agora sim, bloqueia propagação para o pai
            "propagate": True,  # <- sem handler próprio
        },
       
        # "__main__": { #  <- logger chamado quando usa getLogger(__name__)
        #     "handlers": ["console"],
        #     "level": "DEBUG",
        #     "propagate": False,
        # },
        # "root": {  #  <-  logger raiz
        #     "handlers": ["console"],
        #     "level": "DEBUG",
        # },
    },
}

import time
def print_with_delay(words, delay = 0.01):
    """
    Imprime palavras com delays personalizados, tudo na mesma linha.

    Args:
        words (list | str): Lista de palavras ou string com palavras separadas por espaço.
        delay (float): valor para dar de delay entre cada letra.
                      Se não for fornecida, o delay padrão é 0.05 segundos.
    """
    if isinstance(words, str):
        words = words.split(" ")
        
    for word in words:
        for character in word:
            print(character, end="", flush=True)
            time.sleep(delay)
        print(" ", end="", flush=True)
    print()
