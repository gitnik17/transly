.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity
   :alt: Maintenance

.. image:: https://img.shields.io/badge/python-3.above-blue.svg
   :target: https://img.shields.io/badge/python-3.above-blue.svg
   :alt: versions


Transly
=======
Transly is a sequence to sequence Bi-directional LSTM Encoder-Decoder model that's trained on the
`CMU pronouncing dictionary`_, `IIT Bombay English-Hindi Parallel Corpus`_
and `IIT Kharagpur transliteration corpus`_.

.. _CMU pronouncing dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
.. _IIT Bombay English-Hindi Parallel Corpus: http://www.cfilt.iitb.ac.in/iitb_parallel/
.. _IIT Kharagpur transliteration corpus: https://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/index.html

The *pronunciation module* in Transly can predict pronunciation of any given word *(with an American accent of course!)*

Take any word of any language - just transliterate the word in English (all capitals) and you are good to go.
Be it a new or old, seen or unseen, sensible or insensible word - *Transly can catch'em all!*

Another module in Transly is the *transliteration module*.
It currently supports Hindi to English and English to Hindi transliterations.

Pre-trained models can be found inside the respective trained_models folders. New models can also be trained on custom data.

Installation
============
Use the package manager `pip`_ to install transly

.. _pip: https://pip.pypa.io/en/stable/

.. code-block:: sh

    pip install transly


Usage
=====

Pronunciation
==============
Using the pre-trained pronunciation model

.. code-block:: python

    import transly.pronunciation as tp

    # let's try a hindi word
    # the prediction accent would be American
    QUERY = 'MAKAAN'
    a = tp.load_model(model_path='cmu')
    a.infer(QUERY, separator=" ")

**Training a new model on custom data**

.. code-block:: python

    from transly.seq2seq.config import SConfig
    from transly.seq2seq.version0 import Seq2Seq

    config = SConfig(training_data_path=training_data_path, input_mode='character_level', output_mode='word_level')
    s2s = Seq2Seq(config)
    s2s.fit()
    s2s.save_model(path_to_model=model_path, model_file_name=model_file_name)

Transliteration
===============
Hindi to English
----------------
Using the pre-trained model

.. code-block:: python

    import transly.transliteration as tl

    QUERY = 'नहीं'
    a = tl.load_model(model_path='hi2en')
    a.infer(QUERY)


English to Hindi
----------------
Using the pre-trained model

.. code-block:: python

    import transly.transliteration as tl

    QUERY = 'NAHI'
    a = tl.load_model(model_path='en2hi')
    a.infer(QUERY)


**Training a new model on custom data**

.. code-block:: python

    from transly.seq2seq.config import SConfig
    from transly.seq2seq.version0 import Seq2Seq

    config = SConfig(training_data_path=training_data_path)
    s2s = Seq2Seq(config)
    s2s.fit()
    s2s.save_model(path_to_model=model_path, model_file_name=model_file_name)


Training data file should be a csv with two columns, the input and the output

========  ===============
  Input     Output
========  ===============
   AA           AA1
 AABERG     AA1 B ER0 G
 AACHEN     AA1 K AH0 N
AACHENER  AA1 K AH0 N ER0
========  ===============

License
=======
The Python code in this module is distributed with Apache License 2.0