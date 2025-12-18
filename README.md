# Hackweek 2025 project

## References

Title: **openqa needles management with AI tools**  
Link: https://hackweek.opensuse.org/25/projects/openqa-tests-needles-elaboration-using-ai-image-recognition  
Repository name: openqa-needles-AI-driven  
Author: Maurizio Dati  
Date start: 1/12/2025  

## Description

This project aims to begin analyzing and possibly implementing a first POC for migrating OpenQA's needle management, currently built using numerous images and target object references, to an AI-based framework. The goal is to modernize and improve OpenSUSE testing with more efficient and high-performance tools, reducing maintenance and costs, and improving final quality.

## Structure

**Documents**:   
in the folder `docs/` contains the documentation and project informations to address the coding, the data preparation and training and testing, for a POC. The document [PLAN](docs/PLAN.md) contains the project study and the main steps for the implementation. It contains some code examples too, also available as files in the subdirectory `docs/snippets`.

**Project**:  
The folder `project/` is dedicated to the code and implementation of the project elaboration.

**Data**:  
The folder `data/` is dedicated to the datasets needed for preparing, training and testing the project. Here:  
`datasource`: to store the initial selection of files, for the training preparation.  
`yolo_project`: the root folder for the training data.  


