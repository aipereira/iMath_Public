UMA OptLearn Algorithm implementation

In this document, we will discuss the key aspects of the Python code
that has been developed to support and extend the Dash application for
taking tests, particularly important for our research. The code also
includes the core functionality for a recommender system using the
Surprise library, the main feature of our approach to implement a
OptLearn Algorithm.

Dependencies

Although is recommended just rely on the requirements.txt files that we
use in the Deployment section further down in this document, the
dependencies are the following:

-   dash
-   pandas
-   numpy
-   dash_bootstrap_components
-   waitress
-   random
-   pygsheets
-   time
-   re
-   surprise

Email Verification and Data Integrity

Within the Dash application, logic and code have been added to perform
an email check before allowing users to take a test. This email
verification step is crucial for ensuring the integrity of the data and
avoiding the generation of a flawed dataset. Data quality check is vital
for the success of the project, only with a good set of data will we be
able to make the right decisions to guide learners.

In order to achieve this goal, the main app.layout() was modified to add
a dbc.Container() to alert the user that the email address he/she has
entered has not been previously used. This approach was taken as it was
not the first algorithm to be implemented, so in this simple way we
could make sure that students repeating the questionnaire could be
alerted that they entered a different email address (something common in
the data that was available in the Google Sheet).

Input Validation with Regular Expressions

Another important aspect of the code is the addition of logic for
ensuring that users input all required fields correctly. This involves
parsing user inputs using regular expressions to validate their
correctness. This validation step guarantees that the data collected is
accurate and can be used effectively for subsequent processes.

To get this working, a new @app.callback was defined with the function
check_validity(text, new_user, n_clicks) that updates the properties of
the “email” and “alert” components in response to user interactions. It
is triggered by changes in the “email,” “radio,” and “start_button”
inputs. The function’s behavior is as follows:

-   If the “start_button” is clicked, both the “email” and “alert”
    components are set to False.
-   If the “email” is not empty and matches a specified regular
    expression:
    -   If the user is new (new_user is True), both the “email” and
        “alert” components are set to True.
    -   If the user is not new (new_user is False) and there is no
        matching email in a DataFrame (df3), both the “email” and
        “alert” components are set to True.
-   If none of the above conditions are met, both the “email” and
    “alert” components are set to False.

This provides us a way to update the Dash view and the visibility of the
elements according to the data entered by the user.

Function to Evaluate Student Answers

The function evaluate_answer(question, answer, df) is used to evaluate a
student’s answer based on the difficulty of the question and the correct
answer. This function is key to the recommender system’s ability to
perform collaborative filtering and find similar users. It plays a
crucial role in providing personalized recommendations to users based on
their performance and the questions they answer, since it provides the
rating aforementioned. A more technical insight of the rating can be
found in our Technical Report.

The function is designed to be used with the question id and the answer
that a student provided. The variable df should contain the DataFrame
with the questions and answers that we get when parsing the Google Drive
data, which is explained in the following section.

Data Cleaning and Dataset Generation

To address issues where some users might not have filled in the required
data correctly, code has been added for data cleaning. The resulting
dataset is formatted as follows:

         id          question_id   student_answer   evaluation
  ----------------- ------------- ---------------- ------------
   student1@uma.es       15              1              4
   student1@uma.es       31              3              2
   student1@uma.es        7              2              5
          …               …              …              …
   student2@uma.es       24              4              1
   student2@uma.es        2              1              3
          …               …              …              …

The first step in this data processing workflow involves extracting data
from Google Sheets. Specifically, the process begins by retrieving a
worksheet containing questions. Within this worksheet, we focus on
gathering user-related information, getting the values from the cells
from “D2” containing this data. Then, it also extracts answers-related
data, which commences from cell “I2” and continues to the last row,
encompassing the content within the last column with answer related
data. This process is not just performed in a single Google Sheet, as it
extends to also gather data from our sheet named
“partner_output_Malaga.” Here, the same procedure of extracting both
user and answers data is replicated.

In the next phase, the collected user data from the two Google Sheets is
consolidated into a unified DataFrame. This resulting DataFrame is named
“df_users,” and to ensure consistency and avoid discrepancies, all the
entries are converted to lowercase. Additionally, the answers data is
concatenated into a separate DataFrame, named “df_answers.”

Following the data concatenation, the workflow proceeds to separate the
answers data to format it in a easy way for working with it. This
separation involves creating two distinct DataFrames: “student_answers”
and “test_questions.” The “student_answers” DataFrame stores the
students’ responses, encompassing the data found in odd-numbered
columns, while the “test_questions” DataFrame is reserved for the test
questions, housing the information located in the even-numbered columns.
This removes the hassle to access the columns again and again in the
original data.

The subsequent step involves stacking the data, resulting in the
formation of a new DataFrame that will be the key to work with the data,
df3. In this process, the test questions and student answers are merged
while maintaining a consistent index. This new DataFrame, “df3,” is
endowed with appropriate column names, “question_id” and
“student_answer,” to ensure clarity and ease of access.

To facilitate the mapping of students to their answers, we duplicate the
email addresses found in “df_users”. Each email address is replicated
five times, aligning with the number of responses or answers provided by
each student, since we have five questions per test.

Data cleanliness is imperative, and therefore, a cleaning phase is
introduced. Any rows in “df3” that contain empty values are removed.
This action is taken in response to the presence of empty rows in the
Google Drive documents during the time it was developed.

Lastly, to address any non-standard or erroneous email addresses,
corrections are applied to the “id” column of “df3.” These corrections
ensure that the email addresses conform to the expected format or
standards, enhancing the quality and integrity of the data.

Dataset as Data Cache

The DataFrame df3 is used as a global variable in the code. It also acts
as a data cache, storing all the necessary data at the start of the
program to avoid repeated requests to Google Drive or any other data
source. This optimizes data retrieval and usage, especially in
situations where multiple operations depend on the same dataset. This
provides a way to store all the data from other universities on startup
like seen before. The DataFrame objects of Pandas might cause troubles
when copying as seen in the documentation since it should create a new
object, but this is not the case, so we chose to keep it simple that
way, which also improved our code performance-wise.

Surprise Library Integration

The code includes integration with the Surprise library, which is a
Python library for building and analyzing recommender systems. Surprise
simplifies the process of building and evaluating collaborative
filtering models, a common technique used in recommendation systems.
This library assists in finding similar users and making personalized
recommendations based on user behavior.

Function to Generate Questions

Thanks to our approach using the Surprise library, we defined the
function get_questions(output_row,algo), where we call the Surprise
function to provide question recommendations for students based on their
previous answers and using a specified prediction algorithm. The
function gets the current data of the Dash session with the output_row
variable and the algorithmn we want to use with the algo variable and
starts by loading data from a DataFrame, configuring it with a rating
scale, and then extracting relevant information about the student and
the questions. It identifies questions that the student has already
answered and categorizes them based on the number of times they were
answered. Up to five remaining unanswered questions are selected as
candidates for recommendations. The choice of prediction algorithm, such
as SVD++ or KNNWithMeans, is determined based on the provided “algo”
parameter. This way we could compare the performance of two different
algorithms, each one with a different philosphy: one is based in matrix
factorization and other one in k-NN inspired algorithms. The function
then trains the selected algorithm using the entire dataset and
generates recommendations by predicting student ratings for the
remaining questions. The recommendations are sorted by predicted rating,
and the top five recommended questions are returned. This function is
designed for educational contexts to provide students with personalized
question recommendations based on their historical responses.

------------------------------------------------------------------------

Deployment

The code is located in the “CODE” folder.

For installing the dependencies, two requirements txt files are
provided:

-   pip_requirements.txt
-   conda_requirements.txt

Currently, the website is launched in the following line:

    app.run_server(debug=False,host='127.0.0.1', port=8050,use_reloader=False)

Someone in charge of deploying the code might want to change the host
and port in this line of code.

Conda

For installing the dependencies in conda you can use the following
command:

    conda create --name <environment name > --file conda_requirements.txt

Pip

For installing the dependencies in pip you can use the following
command:

    pip install -r pip_requirements.txt 

------------------------------------------------------------------------

In both cases, you can run the python script with

    python3 question_generator_partner.py