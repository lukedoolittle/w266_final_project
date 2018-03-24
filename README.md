# Question and Answer Generator

The model is based on the design described in [this paper](https://arxiv.org/pdf/1706.01450.pdf) and the initial [project proposal](https://docs.google.com/document/d/1VYZ7kXDQxtpvXVGGQX1S2Q7SgyMtEMuIkSm0S468Lf0/edit) describes the differences.

An updated report is available [here](https://docs.google.com/document/d/1RwLXYNd0J-qpwddoGeKAKEWJ53vJXVYvsT1X2h90Uls/edit?usp=sharing)

## Running

### Option 1

Use the amazon Deep Learning AMI (Ubuntu) Version 5.0

```bash
source activate tensorflow_p36
git clone https://github.com/lukedoolittle/w266_final_project.git
cd w266_final_project
pip install -r requirements.txt
wget -P data/ https://s3.amazonaws.com/projectburton/glove.6B.300d.txt
```

### Option 2

Use the w266 course GCP cloud setup (Anaconda is already installed)

```bash
conda create --name qamodel tensorflow
source activate qamodel
git clone https://github.com/lukedoolittle/w266_final_project.git
cd w266_final_project
pip install -r requirements.txt
wget -P data/ https://s3.amazonaws.com/projectburton/embeddings/glove.6B.100d.txt
```

NOTE: the requirements.txt doesn't list Tensorflow despite that being required to run the project. If the environment doesn't have a native Tensorflow installation then you can install it using pip.

### Training

```bash
wget -P data/ https://s3.amazonaws.com/projectburton/cnn_data/train.csv
python src/runner.py --mode=train --epochs 5 --maxbatchsize 32
```

### Testing

```bash
wget -P data/ https://s3.amazonaws.com/projectburton/cnn_data/test.csv
wget -P src/model/ https://s3.amazonaws.com/projectburton/basic_model/model-5.data-00000-of-00001
wget -P src/model/ https://s3.amazonaws.com/projectburton/basic_model/model-5.index
wget -P src/model/ https://s3.amazonaws.com/projectburton/basic_model/model-5.meta
python src/runner.py --mode=test
```

## Data

For the training data the answers must come directly from the text, so the format is a csv file with a text block, a question, and then a pointer to the answer within the text in the form of a range of indicies. The header row should be as follows: `story_id,story_text,question,answer_token_ranges`

### Single Training Example

`./cnn/stories/644a3f79470d3b457efacc7d4ea33577d59e69c1.story,"WASHINGTON -LRB- CNN -RRB- -- One of the Marines shown in a famous World War II photograph raising the U.S. flag on Iwo Jima was posthumously awarded a certificate of U.S. citizenship on Tuesday . The Marine Corps War Memorial in Virginia depicts Strank and five others raising a flag on Iwo Jima . Sgt. Michael Strank , who was born in Czechoslovakia and came to the United States when he was 3 , derived U.S. citizenship when his father was naturalized in 1935 . However , U.S. Citizenship and Immigration Services recently discovered that Strank never was given citizenship papers . At a ceremony Tuesday at the Marine Corps Memorial -- which depicts the flag-raising -- in Arlington , Virginia , a certificate of citizenship was presented to Strank 's younger sister , Mary Pero . Strank and five other men became national icons when an Associated Press photographer captured the image of them planting an American flag on top of Mount Suribachi on February 23 , 1945 . Strank was killed in action on the island on March 1 , 1945 , less than a month before the battle between Japanese and U.S. forces there ended . Jonathan Scharfen , the acting director of CIS , presented the citizenship certificate Tuesday . He hailed Strank as `` a true American hero and a wonderful example of the remarkable contribution and sacrifices that immigrants have made to our great republic throughout its history . ''",What war was the Iwo Jima battle a part of ?,13:16`

### Downloadable Training Sets

* CNN (preformatted): [Train](https://s3.amazonaws.com/projectburton/train.csv) (350MB), [Test](https://s3.amazonaws.com/projectburton/test.csv) (18MB), [Dev](https://s3.amazonaws.com/projectburton/dev.csv) (18MB)
* SQuAD: [Train](https://s3.amazonaws.com/projectburton/train-v1.1.json) (28.9MB), [Dev](https://s3.amazonaws.com/projectburton/dev-v1.1.json) (4.6MB)
* MARCO: [Train](https://s3.amazonaws.com/projectburton/marco_data/train_v2.0.json) (3.39GB), [TrainAnswers](https://s3.amazonaws.com/projectburton/train_v2.0_well_formed.json) (551.9MB), [Dev](https://s3.amazonaws.com/projectburton/dev_v2.0.json) (423.7MB), [DevAnswers](https://s3.amazonaws.com/projectburton/dev_v2.0_well_formed.json) (68.9MB), [Eval](https://s3.amazonaws.com/projectburton/eval_v2.0.json) (402.9MB), [EvalAnswers](https://s3.amazonaws.com/projectburton/evalpublicwellformed.json) (63.7MB)

## Embeddings

* Glove [100 dim](https://s3.amazonaws.com/projectburton/glove.6B.100d.txt) (331MB)