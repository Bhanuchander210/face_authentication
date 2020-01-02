# Face Authentication System
Simple face authentication system


### Work Flow

![work](face_auth.png)


#### Input Directory Structure

```commandline
input_dir
    |__ person_1
            |__ person_1_face_1.png
            |__ person_1_face_2.png
    |__ person_2
            |__ person_2_face_1.png
            |__ person_2_face_2.png
    |__ ...
```

#### Commands

- Create augmented inputDataSet

```commandline
python face_authentication/augmenter.py --input-dir /path/to/inputDir --output-dir /path/to/augmentedOutDir
```


- Creating embeds from the input directory for multiple classes

```commandline
python face_authentication/create_embeds.py --input-dir /path/to/augmentedOutDir --embeds-dir /path/to/embeds/dir
```

- Predict / Authentication check with saved embeds

```commandline
$ python face_authentication/predict.py --test-image /path/to/unauth_image.png --embeds-dir /path/to/embeds/dir

Unauthorized person.

$ python face_authentication/predict.py --test-image /path/to/new_person_1.png --embeds-dir /path/to/embeds/dir

Face authorized as : perseon_1

```

- Evaluating model

```commandline
$ python face_authentication/eval_model.py --test-dir /path/to/testDataSet --embeds-dir /path/to/embeds/dir
```

Output :

```text
class: ABDUL_KALAM, accuracy: 1.000, distances: 0.000-0.391-0.548, total: 13
class: Tariq_Aziz, accuracy: 1.000, distances: 0.000-0.301-0.425, total: 6
class: Trent_Lott, accuracy: 1.000, distances: 0.000-0.401-0.578, total: 16
class: Venus_Williams, accuracy: 1.000, distances: 0.000-0.383-0.485, total: 17
class: Vladimir_Putin, accuracy: 1.000, distances: 0.000-0.404-0.522, total: 49
class: KALPANA_CHAWLA, accuracy: 1.000, distances: 0.000-0.289-0.475, total: 15
class: Tim_Henman, accuracy: 0.947, distances: 0.000-0.485-0.579, total: 19
```