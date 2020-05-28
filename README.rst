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
Transly is trained on the CMU pronouncing dictionary.

The *pronunciation module* in Transly can predict pronunciation of any given word.
Be it a word of any language - just transliterate the word in English (all capital) and you are good to go!
Be it a new/old, seen/unseen, sensible/insensible word, Transly can catch'em all!

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
    QUERY = 'NAHI'
    a = tp.load_model(model_path='cmu')
    a.infer(QUERY)

Training a new model on custom data
Training data file should be a csv with two columns, the input and the output

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


Training a new model on custom data
Training data file should be a csv with two columns, the input and the output

.. code-block:: python

    from transly.seq2seq.config import SConfig
    from transly.seq2seq.version0 import Seq2Seq

    config = SConfig(training_data_path=training_data_path)
    s2s = Seq2Seq(config)
    s2s.fit()
    s2s.save_model(path_to_model=model_path, model_file_name=model_file_name)

License
=======
The Python code in this module is distributed with Apache License 2.0