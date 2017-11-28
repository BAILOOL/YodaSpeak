# YodaSpeak
Translating English to Yoda English using Sequence-to-Sequence with Tensorflow.

Follow the link for online [demo](http://yodaspeakseq2seq.info/).

## Why
Tensorflow team has provided an awesome [seq2seq tutorial](https://www.tensorflow.org/versions/r1.3/tutorials/seq2seq) on neural network to translate English to French. Though, when I've trained their network, I've had no idea how well it performs in terms of translation from a human perspective since I do not speak French. So, I've decided to perform translation on something that is apprehendable from both sides. Subsequently, [YodaSpeakSeq2Seq](http://yodaspeakseq2seq.info/) appeared.

## Selected examples of translation
|Input|Output|
|-----|------|
| The force is strong with you!    |   Strong with you , the force is !   |
| I feel the presence of dark force!    | Of honorary force I feel the presence . Yes , hmmm .     |
| How much is used car? | Used used , how much is , hmm ? |
| Who is the president of United States? | The president of united states , who is , hmm ? |
| What time is it now? | It now , what time is , hmm ? Hmmmmmm . |
| Do you want to come to my house tonight? | Go do to my trek trek do you , hmm ? |
| You must have patience! | Fun , you must have ! |
| You should die    | Be die , you should .     |
| Yoda is not trained enough.| Enough enough is enough not . |

## Comments

1. Obviously, as you might have noticed, the translation is not working perfectly. I have used only about 200k sentences for the training. It should be better once more data is fed.

2. The server is not made for production. Thus, might fail regularly.

## How to run locally
Clone the repo to your local disk:
```
git clone https://github.com/BAILOOL/YodaSpeak.git
cd YodaSpeak
```

Download the pretrained [model](https://1drv.ms/u/s!AiL1Yzy0p5Yhgxr4g5U_98J5VQvZ) and extract the content:
```
tar -xzvf YodaModels.tar.gz Models
```

Install Flask:
```
pip install Flask
```

Run server locally:
```
python translation_server.py
```

## How to train
Since seq2seq tutorial already provides all the needed codes, the only remaining thing for us is to collect the data. Here, I list what I have done to get English-to-Yoda translation matches:

1. Download English-to-French translation data from the [WMT'15](http://www.statmt.org/wmt15/translation-task.html).
2. Disregarded French part of the data.
3. Feed English sentences to the original [YodaSpeak](http://www.yodaspeak.co.uk/index.php) to get the Yoda ground truth (mainly because I am lazy to reinvent the wheel and make my own translator). API can be found at [Mashape](https://market.mashape.com/explore).
4. Train the network using codes from the tutorial.



