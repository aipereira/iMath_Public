# HAW OptLearn Algorithm implementation

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
