# Natural Language Processing on Movies

This is the final project for CMPT 353 Data Science course. The objective is to determine the success of a movie in terms of popularity by its plot summary. The workflow of the data analysis consists of two main parts: predict movie genres from movie descriptions and analyze the trends of movie genres over time. This allows one to classify the genres of a movie and correlate them to the current trend to see if the identified genres are popular.

## Instructions

Install the libraries:

    pip install --user numpy pandas matplotlib scikit-learn

Clone the repository and run the command in project's root directory:

    python movie-analysis.py

## Folder Structure

    .
    ├── data
    ├── doc
    ├── output
    ├── scripts
    ├── movies_analysis.py
    └── README.md

* The `data` folder contains data pulled from movie review websites.

* The `doc` folder contains project requirements and analysis reports.

* The `scripts` folder contains scripts used to obtain the data. The scripts are provided and not written by me.

* The `output` folder contains processed datasets, which are 4 figures. Text analysis is printed on the console.
