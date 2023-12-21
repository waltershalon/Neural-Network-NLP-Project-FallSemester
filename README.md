# Objectives

The learning objectives of this assignment are to:
1. build a neural network for a community task 
2. practice tuning model hyper-parameters

# Read about the CodaBench Competition

You will be participating in a class-wide competition.
The competition website is:

https://www.codabench.org/competitions/1557/?secret_key=5e170c5e-5ea3-4ddb-a3c2-401e50a7fc80

You should visit that site and read all the details of the competition, which
include the task definition, how your model will be evaluated, the format of
submissions to the competition, etc.

# Create a CodaBench account

You must create a CodaBench account and join the competition:
1. Visit the competition website.

2. In the upper right corner of the page, you should see a "Sign Up" button.
Click that and go through the process to create an account.
**Please use your @arizona.edu account when signing up.**
Your username will be displayed publicly on a leaderboard showing everyone's
scores.
**If you wish to remain anonymous, please select a username that does not reveal
your identity.**
Your instructor will still be able to match your score with your name via your
email address, but your email address will not be visible to other students. 

3. Return to the competition website and click the "My Submissions" tab, accept
the terms and conditions of the competition, and register for the task.

4. Wait for your instructor to manually approve your request.
This may take a few days.

5. You should then be able to return to the "My Submissions" tab and see a
"Submission upload" form.
That means you are fully registered for the task.

# Clone the repository

Clone the repository created by GitHub Classroom to your local machine:
```
git clone https://github.com/ua-ista-457/graduate-project-<your-username>.git
```
You are now ready to begin working on the assignment.

# Download the data

Go to the "Get Started" tab on the CodaBench site, and click on the "Files"
sub-tab.
You should see a button to download the training and development data for the
task.
Download and unzip that data into your cloned repository directory.
Please **do not commit the data to the repository**.

When the test phase of the competition begins, you may return to the "Files"
tab to download the unlabeled test data for the task.

# Write your code

You should design a neural network model to perform the task described on the
CodaBench site.
Your code should train a model on the provided training data and tune
hyper-parameters on the provided development data.
Your code should be able to make predictions on either the development data
or the test data.
Your code should package up its predictions in a `submission.zip` file,
following the formatting instructions on CodaBench.

You must create and train your neural network in the Keras framework that we
have been using in the class.
You should train and tune your model using the training and development data
that you downloaded from the CodaBench site.

**If you would like to use any additional resource to train your model, you must
first ask for permission in the `#programming` channel on the class's Slack
workspace.**

There is some sample code in this repository from which you could start.
This code is described briefly on the CodaBench site.
You should feel free to delete this code entirely and start from scratch if
you prefer.

# Test your model predictions on the development set

During the development phase of the competition, the CodaBench site will expect
predictions on the development set.
To test the performance of your model, run your model on the development data,
format your model predictions as instructed on the CodaBench site, and upload
your model's predictions on the "My Submissions" tab of the CodaBench site.

During the development phase, you are allowed to upload predictions many times.
You are **strongly** encouraged to upload your model's development set
predictions to CodaBench after every significant change to your code to make
sure you have all the formatting correct.

Note that [sometimes submissions on CodaBench sometimes get stuck](https://github.com/codalab/codabench/issues/1184).
If you find that your submission is not scored within 1 minute, cancel the
submission and re-submit it.

# Submit your model predictions on the test set

When the test phase of the competition begins (consult the CodaBench site for
the exact timing), the instructor will update the CodaBench site to expect
predictions on the test set, rather than predictions on the development set.
The instructor will also release the unlabeled test set on CodaBench as
described above under "Download the Data".
To test the performance of your model, download the test data, run your model on
the test data, format your model predictions as instructed on the CodaBench
site, and upload your model's predictions on the "My Submissions" tab of the
CodaBench site.

During the test phase, you are allowed to upload predictions only once.
This is why it is critical to debug any formatting problems during the
development phase.
 
# Grading

You will be graded first by your model's accuracy, and second on how well your
model ranks in the competition.
If your model achieves better accuracy on the test set than the baseline model
included in this repository, you will get at least a B.
If your model achieves better accuracy on the test set than another baseline
that the instructor will reveal after the competition, you will get an A.
All models within the same letter grade will be distributed evenly across the
range, based on their rank.
So for example, the highest ranked model in the A range will get 100%, and the
lowest ranked model in the B range will get 80%.

**If you train your neural network with any library other than Tensorflow/Keras,
or you use an external resource that you do not obtain permission for in the
`#programming` channel of the course's Slack workspace, you will lose 10% (a
letter grade) from your score.**
