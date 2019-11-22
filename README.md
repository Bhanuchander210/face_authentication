# Face Authentication System
Simple face authentication system



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

- Creating embeds from the input directory for multiple classes

```commandline
python face_authentication/create_embeds.py --input-dir /path/to/input/classes --embeds-dir /path/to/embeds/dir
```

- Predict / Authentication check with saved embeds

```commandline
$ python face_authentication/predict.py --test-image /path/to/unauth_image.png --embeds-dir /path/to/embeds/dir

Unauthorized person.

$ python face_authentication/predict.py --test-image /path/to/new_person_1.png --embeds-dir /path/to/embeds/dir

Face authorized as : perseon_1

```
