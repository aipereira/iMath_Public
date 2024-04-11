# HAW OptLearn Algorithm implementation

In this readme file, we will go through the key parts of the Python code that has been written to support and enhance the Dash application for taking tests, which is especially relevant for our study. Our code also contains the fundamental functionality for a recommender system built with the Surprise module, which is the major element of our approach to implementing an OptLearn Algorithm.

## Extended implementation from UMA

The provided code is an extended, refactored, and improved version developed by UMA, so it's strongly recommended that the readme from that implementation is read to give the reader the background and knowledge from parts omitted or taken for granted in this readme file.

## Dependencies

Although is recommended just rely on the requirements.txt files that we use in the Deployment section further down in this document, the dependencies are the following:

* dash
* pandas
* numpy
* dash_bootstrap_components
* waitress
* random
* pygsheets
* time
* re
* surprise

## Function loadStudentAnswers()

The `loadStudentAnswers` function is responsible for loading and consolidating student answers from multiple sheets or files, either from Google Sheets using pygsheets or from Excel files using pandas. It accepts the endpoint, a list of sheet names, and an optional flag to determine the data source. For each sheet provided, the function extracts the relevant data columns, which include the student's email, questions, and their corresponding answers. The extracted data is stored in separate DataFrames. After processing all the sheets, these DataFrames are concatenated into a single DataFrame named df_answers, ensuring consistent column structure. The number of questions and answers is determined based on the DataFrame's column count, and appropriate column names are generated. The function ultimately returns a DataFrame with standardized columns, making it easier to work with the collected student answers. This function is essential for consolidating data from multiple sources in educational or survey contexts.

The resulting DataFrame has the following structure: 

|        **Email**        | **Question1** | **Answer1** | **…** | **Answer4** | **Question5** | **Answer5** |
|:-----------------------:|:-------------:|:-----------:|:-----:|:-----------:|:-------------:|:-----------:|
| student1@haw-hamburg.de | 21            | 1           |   …   | 1           | 33            | 2           |
| student2@haw-hamburg.de | 8             | 0           |   …   | 1           | 11            | 1           |
| student3@haw-hamburg.de | 11            | 3           |   …   | 3           | 23            | 0           |
| student4@haw-hamburg.de | 9             | 2           |   …   | 1           | 19            | 3           |
|            …            |       …       |      …      |   …   |      …      |       …       |      …      |

The DataFrame is organized such that for each test, identified by the student's email address, the subsequent columns are arranged in pairs. These pairs consist of a "QuestionX" column, where X represents the number of the question presented to the student, and the corresponding "AnswerX" column, which contains the student's response. Each pair of columns represents a specific question-answer pair for that student. The "QuestionX" column holds the question number, while the "AnswerX" column holds the value corresponding to the student's answer for that question. This structure continues for each student, allowing for a clear and systematic representation of their responses to a series of questions.

This DataFrame allows us to efficiently link each student's test responses to their respective questions and answers, while also having a nicer way to work with the data. 

## Function generateRatingDataset()

The `generateRatingDataset` function is a Python function designed to take an input DataFrame (`df_answers`) and perform a series of operations to transform the data into a new DataFrame. This function is designed for data processing and likely for the purpose of rating or analyzing responses. 

The initial step involves extracting specific columns from the input DataFrame. It first identifies the 'Email' column, which likely contains email addresses associated with the respondents. It then identifies pairs of columns representing questions and their corresponding answers in the input DataFrame. 

Next, the function constructs a new DataFrame (`new_df`) with three columns:
1. 'email': This column is created by repeating the 'Email' values to match the length of the flattened question and answer columns. Each email address is associated with multiple rows of questions and answers, preserving the linkage to the original students.
2. 'question': This column contains the flattened values of the original question columns from the input DataFrame. It consolidates all the questions into a single column.
3. 'answer': Similarly, this column contains the flattened values of the corresponding answers from the input DataFrame. All answers are placed in a single column, maintaining their alignment with the respective questions.

The final step is returning this newly structured DataFrame, which is now ready for further analysis, rating (which is what we were looking for in our case), or any other type of processing. The `generateRatingDataset` function streamlines and reorganizes the data, making it convenient for any tasks that involve evaluating or work with responses in a clear and concise format.  

The resulting DataFrame has the following structure: 

|        **email**        | **question** | **answer** |
|:-----------------------:|:------------:|:----------:|
| student1@haw-hamburg.de | 21           | 0          |
| student1@haw-hamburg.de | 7            | 2          | 
| student1@haw-hamburg.de | 8            | 1          | 
| student1@haw-hamburg.de | 33           | 3          |
|           ...           |      ...     |     ...    |


## Function generate_DataFrame()

The generate_DataFrame function manipulates and process data stored in sheets. It begins by defining a list of sheet names and then loads data from these sheets into a DataFrame using the afromentioned function `generateRatingDataset()`. Subsequently, the data is cleaned, with rows containing missing values removed, and email addresses are corrected in the resulting DataFrame. Optionally, the function allows for filtering the data based on specific questions. This allows to any higher-level component to seamlessly interact with our recommender system by specifying a set of questions they desire. It provides an elegant and flexible means to integrate additional logic, such as question filtering, aimed at enhancing the learning pathway, thereby optimizing the learning experience. 

The function also retrieves data from the provided CSV file with levels and merges it with the existing DataFrame, creating a new dataset named `questions_answers`. A global dictionary, `question_dict`, is used to store mappings related to questions with with its level and the correct answer. The function calculates an 'evaluation' score based on the answer of the student and the level of question and returns the created DataFrame. In essence, this function provides a structured way to enrich and clean the data for offering a processed DataFrame for further utilization with the Surprise library.

The resulting DataFrame has the following structure: 

|        **email**        | **question** | **answer** | **evaluation** |
|:-----------------------:|:------------:|:----------:|:--------------:|
| student1@haw-hamburg.de | 21           | 0          | 3,4            |
| student1@haw-hamburg.de | 7            | 2          | 4,2            |
| student1@haw-hamburg.de | 8            | 1          | 5              |
| student1@haw-hamburg.de | 33           | 3          | 1              |
|           ...           |      ...     |     ...    |       ...      |

The 'evaluation' column is computed using a lambda function that iterates through each row of the DataFrame. The underlying logic is closely related to the data within the provided CSV file from IPB, but this logic can be flexibly adjusted to accommodate different question levels. One can easily adapt this function by modifying the lambda function itself or by parameterizing it with the number of levels and fine-tuning the level definitions to match specific criteria. It's crucial to maintain the integrity of the grading hierarchy, where lower values are associated with incorrect responses to straightforward questions, and higher values correspond to correct answers for more challenging questions. This flexibility ensures that the 'evaluation' column remains adaptable to diverse assessment contexts and grading scales. By keeping the values between 1 and 5 we ensure we are using Surprise correctly to gather useful data.

## Function to Generate Questions

The `get_questions()` function has undergone a refactoring process from the UMA version. Additionally, it now employs the KNNBasic algorithm in lieu of KNNWithMeans. This strategic adjustment was made after careful analysis of our data, revealing that the basic algorithm, KNNBasic, is a more fitting choice when assessed in terms of the root mean square error (RMSE). Since it's just a refactored version of the original function, it's advised to read the readme as we said before.

***

## Deployment

For installing the dependencies, two requirements txt files are provided: 

* pip_requirements.txt
* conda_requirements.txt 

Currently, the website is launched in the following line:

	app.run_server(debug=False,host='127.0.0.1', port=8050,use_reloader=False)

Someone in charge of deploying the code might want to change the host and port in this line of code.

## Conda 
For installing the dependencies in conda you can use the following command:

    conda create --name <environment name > --file conda_requirements.txt

## Pip 
For installing the dependencies in pip you can use the following command:

    pip install -r pip_requirements.txt 
***
In both cases, you can run the python script with 

	python3 question_generator_partner.py
