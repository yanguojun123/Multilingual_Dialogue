
# Mixture-of-languages Routing for Multilingual Dialogues

## 1.Introduction
 We consider multilingual dialogue systems and ask how the performance of a dialogue system can be improved by using information that is available in other languages than the language in which a conversation is being conducted. We adopt a collaborative chair-experts framework, where each expert agent can be either monolingual or cross-lingual, and a chair agent follows a mixture-of-experts procedure for globally optimizing multilingual task-oriented dialogue systems. We propose a mixture-of-languages routing framework that includes four functional components, i.e., input embeddings of multilingual dialogues, language model, pairwise alignment between the representation of every two languages, and mixture-of-languages.
 We quantify language characteristics of unity and diversity using a number of similarity metrics, i.e., genetic similarity, and word and sentence similarity based on embeddings. Our main finding is that the performance of multilingual task-oriented dialogue systems can be greatly impacted by three key aspects, i.e., data sufficiency, language
characteristics, and model design in a mixture-of-languages routing framework.

## 2.Installation

The package general requirements are

- Python == 3.7.10
- Pytorch >= 1.10.2 (installation instructions [here](https://pytorch.org/))
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

1- The package can be installed by running the following command.  

```pip install -r requirements.txt```

2- Make some directory to save information.
```
mkdir tensorboard_summary output
mkdir output/result output/model 
```


## 3. Experiment
Basic experimental commands. All subsequent experiments can be run by changing parameters.

 ```bash
  sh run_mt5.sh [GPU_NO.] [directory_name] [exp_name] [python_file] [source_lang] [target_lang] [train_data][eval_data][test_name]'[Optional parameter]'
 ```

**paramaters**:
GPU_NO.: Specify the available GPU number.
directory_name: the name of the directory, show the different settings. Such as Monolingual or Crosslingual.
exp_name: The name of the experiment specified is also used as the name of the log file.
source_lang: the source language, one of the [en, de, it] in DST task or one of the [en, th, es] in NLU task.
target_lang: target language, one of the [en, de, it] in DST task or one of the [en, th, es] in NLU task.
train_data: train dataset(There may be more than one training set. Please separate it with commas).
eval_data: evaluation dataset(only one).
test_data: test dataset (only one).
Optional parameter: batch_size, mode ...(This part of parameter description can view the in the code file setup_train_args function part.)


### 3.1 train

#### 3.1.1 Natural Language Understanding(NLU)

##### 3.1.1.1 data
1). Monolingual data:
      data/nlu_process/nlubiomlt_{lang}_{type}.txt: 
      lang: one of the [en, es, th]
      type: one of the [train, val, test]
2). Crosslingual data:
    nluclcsa0.7_es_train.txt
    nluclcsa0.7_th_train.txt
    

##### 3.1.2.2 settings:
1)**Monolingual**
For example, The following example is an example of English.

```
sh run_mt5.sh gpu06 Monolingual train_nlu nlu_mon_en en en nlubiomlt_en_train.txt nlubiomlt_en_val.txt nlubiomlt_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
```

2)**Bilingual mixture-of-languages routing**
For example, The following example is an example of from English and Spanish to English.

```bash
sh run_mt5.sh gpu06 Multilingual train_nlu nlu_mot_enes2en es en nlubiomlt_en_train.txt,nlubiomlt_es_train.txt nlubiomlt_en_val.txt nlubiomlt_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
```
3)**Multilingual mixture-of-languages routing**
 train(To complete a complete experiment requires two stages of training):

For example, from English to Spanish, we need to use Thai as the intermediate language for the transition.


**step1**:
 ```
 sh run_mt5.sh gpu06 Multilingual train_nlu mt5_mot_esth2th th es nlubiomlt_es_train.txt,nlubiomlt_th_train.txt nlubiomlt_th_val.txt nlubiomlt_th_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
 ```
**step2**:
We need to train on the best results obtained in the first step.
```
sh run_mt5.sh gpu06 Multilingual train_nlu nlu_mot_enes2es es en nlubiomlt_en_train.txt,nlubiomlt_es_train.txt nlubiomlt_en_val.txt nlubiomlt_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train --pretrained_model=/data/best_epoch/'
```

4)**Crossligual**
For example, train on the English data but test on the Thai data.

 ```
 sh run_mt5.sh gpu06 Monolingual train_nlu nlu_clcsa_en2th en th nluclcsa0.7_th_train.txt nlubiomlt_th_val.txt nlubiomlt_th_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
 ```

#### 3.1.2 Dialogue State Tracking(DST)


##### 3.1.2.1 data
1)Monolingual/Bilingual/Multilingual data:
 data/mulwoz_process/beliefinput2delex_{lang}_{type}.txt
 lang: one of the [en, de, it]
 type: one of the [train, val, test]

2)Crosslingual data:
    beliefCOSDA_de_match
    beliefCOSDA_it_match

##### 3.1.2.2 settings:
**1)Monolingual**
For example, The following example is an example of English.

```bash
sh run_mt5.sh gpu06 Monolingual train_dst dst_mon_en en en beliefinput2delex_en_train.txt beliefinput2delex_en_val.txt beliefinput2delex_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
```
**2)Bilingual mixture-of-languages routing**
For example, The following example is an example of from English and German to English.

```bash
sh run_mt5.sh gpu06 Multilingual train_dst dst_mot_ende2en de en beliefinput2delex_en_train.txt,beliefinput2delex_de_train.txt beliefinput2delex_en_val.txt beliefinput2delex_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
```
**3)Multilingual mixture-of-languages routing**
 train(To complete a complete experiment requires two stages of training):

For example, from English to German, we need to use Italian as the intermediate language for the transition.


**step1**:
 ```bash
 sh run_mt5.sh gpu06 Multilingual train_dst dst_mot_deit2it it de beliefinput2delex_de_train.txt,beliefinput2delex_it_train.txt beliefinput2delex_it_val.txt beliefinput2delex_it_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
 ```
**step2**:
We need to train on the best results obtained in the first step.

```bash
sh run_mt5.sh gpu06 Multilingual train_dst dst_mot_ende2de en de beliefinput2delex_en_train.txt,beliefinput2delex_de_train.txt beliefinput2delex_de_val.txt beliefinput2delex_de_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train --pretrained_model=/data/best_epoch/'
```
4)**Crossligual**

For example, train on the English data but test on the Italian data.
 ```
 sh run_mt5.sh gpu06 Monolingual train_dst dst_clcsa_en2it en it beliefCOSDA_it_match.txt beliefinput2delex_it_val.txt beliefinput2delex_it_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train'
 ```


#### 3.1.3 Natural Language Generation(NLG)

##### 3.1.3.1 data
1)Monolingual/Bilingual/Multilingual data:
 data/mulwoz_process/woz_{lang}_{type}_nlg.txt
 lang: one of the [en, de, it]
 type: one of the [train, val, test]

##### 3.1.3.2 settings:
**1)Monolingual**
For example, The following example is an example of English.

```bash
sh run_nlg.sh gpu06 Monolingual  nlg_mon_en train_nlg en en woz_en_train_nlg.json woz_en_dev_nlg.json woz_en_test_nlg.json '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train --evaluate_type=loss'
```
**2)Bilingual mixture-of-languages routing**
For example, The following example is an example of from English and German to English.

```bash
sh run_nlg.sh gpu06 Multilingual nlg_mot_ende2en train_nlg de en woz_en_train_nlg.json,woz_de_train_nlg.json woz_en_dev_nlg.json woz_en_test_nlg.json '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train --evaluate_type=loss'
```
**3)Multilingual mixture-of-languages routing**
 train(To complete a complete experiment requires two stages of training):

For example, from English to German, we need to use Italian as the intermediate language for the transition.


**step1**:
 ```bash
 sh run_nlg.sh gpu06 Multilingual  nlg_mot_deit2it train_nlg it de woz_it_train_nlg.json,woz_de_train_nlg.json woz_de_dev_nlg.json woz_de_test_nlg.json '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train --evaluate_type=loss'
 ```
**step2**:
We need to train on the best results obtained in the first step.

```bash
sh run_nlg.sh gpu06 Multilingual  nlg_mot_ende2de train_nlg en de woz_en_train_nlg.json,woz_de_train_nlg.json woz_de_dev_nlg.json woz_de_test_nlg.json '--batch_size=6 --gradient_accumulation=2 --prefix --mode=train --pretrained_model=/data/best_epoch/ --evaluate_type=loss'
```

### 3.2 evaluation

Use the same command as training, but you need to change some parameters. Specifically, you need to add the model folder(--dialogue_model_output_path) to be evaluated and set the mode to be evaluated(--mode=evluation).

For example, we evaluate on the Monlingual English data.
```bash
sh run_mt5.sh gpu06 Monolingual train_dst mt5_mot_en_eval en en beliefinput2delex_en_train.txt beliefinput2delex_en_val.txt beliefinput2delex_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=eva --dialogue_model_output_path=mon_en_model/'
```
### 3.3 test

Use the same command as training, but you need to change some parameters. Specifically, you need to add the model file(--pretrained_model) to test and set the mode to be test.(--mode=test).

For example, we test on the Monlingual English data.
```bash
sh run_mt5.sh gpu06 Monolingual train_dst mt5_mot_en_test en en beliefinput2delex_en_train.txt beliefinput2delex_en_val.txt beliefinput2delex_en_test.txt '--batch_size=6 --gradient_accumulation=2 --prefix --mode=test --pretrained_model=../output/model/mon_en_model/model_epoch4/'
```




## 4.Document 
	1. data: save data
		1.1 mulwoz: original data of dst
		1.2 mulwoz_process: processed data of dst
		1.3 nlu: original data of nlu
		1.4 nlu_process: processed data of nlu
		1.5 vocab: code switching data  
		1.6 ontology: ontology data of original dst data 
	2. Multilingual: for bilingual setting and multilingual setting
	3. Mononligual: for mononlingual setting and crosslingual setting
	4. output: save model and result
		4.1 model: save model
		4.2 result: save test result
	5. tensorboard_summary: Save the data that you can visualize
	6. run_mt5.sh: run bash file
	7. process.py: python file for processing data  


		
## 5.Citation

## 6.License