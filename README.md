# SpineWise
A Desktop Application to Detect Bad Posture and Provide Curated e-commerce suggestions to minimize possible health complications. 
## Features
- Posture Detection
- Alerting System
- Posture Scoring Calculation
- Product Recommendations
- Secure and Reliable Privacy Policy
- User-Focused and Clean UI
- Open Source
## Screenshots of Project Progress

## Github Developers Instructions:
1. Clone the Repo on your local machine
```bash
git clone https://github.com/<your-team>/<repo-name>.git
cd <repo-name>
```
2. Fetch all branches
```bash
git fetch --all
```
3. Check out the dev branch
```bash
git checkout dev
git pull origin dev
```
4. Then create a personal branch feature/initials-taskName-title
```bash
git checkout -b feature/initials-taskName-title
git push -u origin feature/initials-taskName-title
```
5. After the code is done, push the branch
```bash
git add .
git commit -m "Describe what you implemented"
git push
```
6. Create a pull request into the dev branch
- Go to Github
- Click "Compare and Pull Request"
- Set the base branch to `dev`
- Add a clear title + short description
7. Once it is ready and dev members review and approve, merge it into the dev branch
  - Do not merge into `main` branch
  - `dev` is for combining features during the sprint

## Installing OpenCV on your local machine instructions
### Instructions for MacBook, assuming you already have Homebrew, Python, and pip installed
1. In terminal run:
```bash
pip3 install opencv-python
```
  - When you do this step, you may run into an error called externally-managed environment. Here is how to fix that error:
  - 1. Create a new virtual environment:
    ```bash
      python3 -m venv opencv-env
    ```
2. Install OpenCV using pip install:
  ```bash
   pip install opencv-python
  ```
- You can update to the latest version of pip with the command:
  ```bash
  pip install --upgrade pip
  ```
3. Check if it was installed and look at the version number using:
  ```bash
   print(cv2.__version__)
  ```
## Installing MediaPipe on your local machine instructions
### Instructions for MacBook, assuming you have an active Python environment, and you have created an OpenCV project in VSCode.
1. Ensure that your virtual environment is activated
```bash
source opencv-env/bin/activate
```
2. Install MediaPipe
```bash
pip install mediapipe
```
3. Import mediapipe in your Python file
```bash
import mediapipe as mp
```
4. Since we are going to be tracking the body, we should import the ``mp_pose.Pose()`` for body tracking






