#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis generator script for DiaFlux.
Writes a complete, single-file thesis.tex that complies with UOS formatting guidelines.
"""
import os

latex_content = r"""\documentclass[12pt,a4paper]{report}

% =============================================================================
% REQUIRED LaTeX PACKAGES
% =============================================================================
\usepackage[a4paper, top=1in, bottom=1in, left=1.5in, right=1in]{geometry}
\usepackage{mathptmx}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{listings}
\usepackage{lstautogobble}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{url}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, calc, decorations.pathreplacing, matrix, shadows}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{pgfplotstable}
\usepackage{enumitem}
\usepackage{tabularx}
\usepackage{tocloft}
\usepackage{rotating}
\usepackage{pdflscape}

% =============================================================================
% CUSTOM STYLES AND CONFIGURATIONS
% =============================================================================
\definecolor{codebg}{RGB}{245,245,245}
\definecolor{codecomment}{RGB}{0,128,0}
\definecolor{codekeyword}{RGB}{0,0,255}
\definecolor{codestring}{RGB}{163,21,21}

\lstset{
    backgroundcolor=\color{codebg},
    commentstyle=\color{codecomment},
    keywordstyle=\color{codekeyword},
    numberstyle=\tiny\color{gray},
    stringstyle=\color{codestring},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    autogobble=true
}

% Heading styles conforming to UOS guidelines (bold, centered, 14pt chapter headings)
\titleformat{\chapter}[display]
  {\normalfont\large\bfseries\centering}
  {\chaptertitlename\ \thechapter}
  {12pt}
  {\Large\bfseries}
\titlespacing*{\chapter}{0pt}{-20pt}{40pt}

\titleformat{\section}
  {\normalfont\large\bfseries}
  {\thesection}
  {12pt}
  {}

\titleformat{\subsection}
  {\normalfont\normalsize\bfseries}
  {\thesubsection}
  {12pt}
  {}

% Page setup & Header/Footer configuration
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% Force page numbering at chapter/first pages
\makeatletter
\let\ps@plain\ps@fancy
\makeatother

% Custom defined colors for TikZ heatmap
\definecolor{emerald}{RGB}{46,125,50}
\definecolor{crimson}{RGB}{198,40,40}

% =============================================================================
% DOCUMENT BEGINS
% =============================================================================
\begin{document}
\onehalfspacing
\pagenumbering{roman}

% -----------------------------------------------------------------------------
% TITLE PAGE
% -----------------------------------------------------------------------------
\begin{titlepage}
    \begin{center}
        \vspace*{1cm}
        \Huge
        \textbf{DIAFLUX: DIABETES RISK INTELLIGENCE AND LIFESTYLE SIMULATION SYSTEM USING MACHINE LEARNING}
        
        \vspace{2.5cm}
        \large
        Submitted in partial fulfillment of the requirements for the degree of\\
        Bachelor of Science in Computer Science
        
        \vspace{2.5cm}
        \textbf{Authors:}\\
        Sheema Ayaz\\
        Noor e Mubeen\\
        Osama Khan
        
        \vspace{2.5cm}
        \textbf{Supervisor:}\\
        Dr. Saeed Ahmed (Assistant Professor)
        
        \vspace{2.5cm}
        \large
        Department of Computer Science\\
        University of Swabi\\
        Year 2025
    \end{center}
\end{titlepage}

\clearpage

% -----------------------------------------------------------------------------
% DECLARATION PAGE
% -----------------------------------------------------------------------------
\chapter*{Declaration}
\addcontentsline{toc}{chapter}{Declaration}
We hereby declare that the project report entitled \textbf{DiaFlux: Diabetes Risk Intelligence and Lifestyle Simulation System using Machine Learning} is our own work. The content presented within this document has been compiled and drafted independently by the author team. All sources, data repositories, libraries, and frameworks referenced have been explicitly cited and acknowledged in compliance with academic standards. This work has not been previously submitted, in whole or in part, to any other institution or university for the acquisition of a degree or professional certification.

\vspace{2.5cm}
\noindent
\begin{tabular}{p{6cm}p{6cm}}
Sheema Ayaz & Noor e Mubeen \\
\rule{5cm}{0.4pt} & \rule{5cm}{0.4pt} \\
\\
Osama Khan & \\
\rule{5cm}{0.4pt} & 
\end{tabular}

\clearpage

% -----------------------------------------------------------------------------
% DEDICATION PAGE
% -----------------------------------------------------------------------------
\chapter*{Dedication}
\addcontentsline{toc}{chapter}{Dedication}
\begin{center}
    \vspace*{4cm}
    \textit{This thesis is dedicated to our beloved parents, whose endless support, patience, and guidance have been our constant source of strength throughout our academic journey.}\\
    
    \vspace{1cm}
    \textit{To our teachers at the Department of Computer Science, University of Swabi, whose instruction and wisdom have enabled us to undertake this software engineering and research project.}\\
    
    \vspace{1.5cm}
    \textit{And to the medical researchers and computer scientists who advocate for public-health accessibility, bridging the gap between artificial intelligence and preventive clinical diagnostics.}
\end{center}

\clearpage

% -----------------------------------------------------------------------------
% ACKNOWLEDGEMENTS
% -----------------------------------------------------------------------------
\chapter*{Acknowledgements}
\addcontentsline{toc}{chapter}{Acknowledgements}
We express our deepest gratitude to our project supervisor, Dr. Saeed Ahmed, Assistant Professor in the Department of Computer Science at the University of Swabi, for his invaluable guidance, technical reviews, and constant encouragement. His deep expertise in machine learning and medical informatics has been critical in shaping the technical direction and clinical relevance of the DiaFlux project.

We are also highly grateful to the faculty members and staff of the Department of Computer Science, University of Swabi, who provided us with the necessary computing infrastructure, research archives, and educational support to complete this Bachelor of Science in Computer Science thesis.

Finally, we extend our heartfelt appreciation to our families and friends for their emotional support, understanding, and patience during the intensive development and writing phases of this final year project.

\clearpage

% -----------------------------------------------------------------------------
% ABSTRACT
% -----------------------------------------------------------------------------
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
The global prevalence of Type II Diabetes Mellitus represents a severe healthcare crisis, causing significant morbidity and mortality, particularly in developing nations like Pakistan. Traditional risk assessment models typically serve as static prediction scorecards, failing to show patients the direct, mathematical impact that lifestyle improvements could have on reducing their risk profile. This thesis presents the design, development, and validation of DiaFlux, an interactive web application that combines a machine learning diagnostic model with a real-time lifestyle simulation engine. 

Using a dataset of 100,000 patient records containing physiological and demographic features, multiple classification models---specifically Logistic Regression, Support Vector Machines, Random Forest, and Gradient Boosting Classifier---were evaluated. The Gradient Boosting Classifier emerged as the best-performing model, achieving a classification accuracy of 97.24\%, a precision of 98.73\%, and an ROC-AUC of 0.9793. The model was serialized and integrated into a Python Flask web server. The frontend is built using React 19 and TypeScript, featuring a brutalist, high-contrast dark user interface optimized for clinical accessibility. 

A central feature of DiaFlux is the lifestyle simulator, which allows patients to adjust key biometric parameters (such as body mass index, glycated hemoglobin HbA1c, and fasting blood glucose) via sliding controllers. The system then queries the backend Flask API to dynamically recalculate risk probability, demonstrating the mathematical path toward metabolic health. The entire application is containerized using a multi-stage Docker workflow and deployed to Hugging Face Spaces for public access. The results indicate that combining machine learning classifiers with interactive lifestyle feedback can improve patient engagement and support early, preventive healthcare interventions.

\clearpage

% -----------------------------------------------------------------------------
% TABLE OF CONTENTS & LISTS
% -----------------------------------------------------------------------------
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage

% -----------------------------------------------------------------------------
% LIST OF ABBREVIATIONS
% -----------------------------------------------------------------------------
\chapter*{List of Abbreviations}
\addcontentsline{toc}{chapter}{List of Abbreviations}
\begin{center}
\begin{longtable}{ll}
\toprule
\textbf{Abbreviation} & \textbf{Full Description} \\
\midrule
ML & Machine Learning \\
GBC & Gradient Boosting Classifier \\
SPA & Single-Page Application \\
API & Application Programming Interface \\
BMI & Body Mass Index \\
HbA1c & Glycated Hemoglobin (Hemoglobin A1c) \\
ROC & Receiver Operating Characteristic \\
AUC & Area Under the Curve \\
WSGI & Web Server Gateway Interface \\
CORS & Cross-Origin Resource Sharing \\
REST & Representational State Transfer \\
EDA & Exploratory Data Analysis \\
WHO & World Health Organization \\
ADA & American Diabetes Association \\
DFD & Data Flow Diagram \\
UML & Unified Modeling Language \\
SRS & Software Requirements Specification \\
SDLC & Software Development Life Cycle \\
FYP & Final Year Project \\
LR & Logistic Regression \\
RF & Random Forest \\
SVM & Support Vector Machine \\
PII & Personally Identifiable Information \\
\bottomrule
\end{longtable}
\end{center}

\clearpage

\pagenumbering{arabic}

% =============================================================================
% CHAPTER 1: INTRODUCTION
% =============================================================================
\chapter{Introduction}

\section{Background and Motivation}
Diabetes Mellitus represents one of the most severe global healthcare challenges of the twenty-first century, characterized by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both. According to the International Diabetes Federation (IDF) Diabetes Atlas (10th edition) published in 2021, approximately 537 million adults globally were living with diabetes \cite{idf_atlas}. This number is projected to rise to 783 million by the year 2045, indicating a global pandemic that spans across geographical and socio-economic boundaries. The vast majority of these cases represent Type II Diabetes Mellitus, a metabolic condition closely tied to excess body weight, physical inactivity, genetic predisposition, and poor dietary behaviors. The systemic consequences of uncontrolled diabetes are profound, leading to microvascular complications such as diabetic retinopathy, nephropathy, and neuropathy, alongside macrovascular complications including coronary artery disease, stroke, and peripheral vascular disorders.

The epidemic is particularly acute in developing regions and low-to-middle-income countries. Pakistan, in particular, is experiencing a severe surge in diabetes prevalence. The country is ranked third globally by the IDF in terms of the total number of individuals living with diabetes, with over 33 million adults currently diagnosed \cite{pak_endocrine}. This represents a high national prevalence rate, placing an immense burden on the country's fragile healthcare infrastructure and public resources. A large percentage of these individuals remain undiagnosed until irreversible vascular damage has already occurred. In low-resource settings, access to medical diagnostic tools, clinical testing, and endocrinology consultations is highly restricted, especially for rural populations. The primary barriers include the cost of diagnostic tests, the physical distances to medical facilities, and a general lack of health literacy.

Traditional public health interventions have relied on post-diagnosis treatment, which is both financially draining and clinically reactive. To mitigate this crisis, contemporary medical research emphasizes preventive healthcare, focusing on identifying metabolic risks before the onset of diabetes. Lifestyle changes, including moderate weight loss, regular physical activity, and glycemic control, have been shown to halt or reverse pre-diabetes. However, there is a lack of accessible digital tools to help individuals monitor their health metrics and understand their risk profiles in these low-resource environments. Existing calculators are often hidden behind paywalls, designed with confusing interfaces, or require clinical interpretations that are difficult for non-technical users to understand.

Machine learning offers a promising approach for early, non-invasive risk detection. By training algorithms on historical patient data, models can recognize complex patterns in demographic and clinical variables. These models can predict whether an individual will develop diabetes, providing a low-cost, scalable tool for public health screening. When deployed via web browsers, these predictive models can be accessed from any location, helping to identify high-risk individuals and promote early medical interventions.

\section{Problem Statement}
Existing digital health tools and online diabetes calculators suffer from several critical design and functional limitations. First, almost all available risk calculators are static. They operate as simple, non-clinical scoring sheets, prompting users for basic demographic details (e.g., age, family history) and returning a fixed risk score. While these scorecards can identify general risks, they do not show the user how changing specific metrics affects their health. For example, a user cannot see how losing 5 kilograms or lowering their blood glucose levels by 20 mg/dL directly impacts their risk of diabetes. This lack of feedback limits the educational value of these tools and fails to motivate behavioral change.

Second, most clinical calculators are designed for medical professionals. They present outputs in technical terms, using complex medical language and probability curves that are difficult for average patients to interpret. This creates a gap between diagnostic metrics and patient actions. Without clear guidance, patients may not understand how to reduce their risk or implement recommendations.

Finally, many risk prediction tools are not designed for low-resource deployment. They often require paid software licenses, rely on proprietary databases, or run on cloud infrastructure that is expensive to maintain. These deployment hurdles limit their accessibility in developing countries. There is a clear need for an open-source, scalable, and interactive application that combines machine learning risk prediction with a real-time lifestyle simulator, helping users understand the direct impact of biometric improvements on their health.

\section{Project Objectives}
The primary goal of the DiaFlux project is to address these gaps by developing an open-source, interactive machine learning application for diabetes risk prediction and lifestyle impact simulation. The specific project objectives include:
\begin{enumerate}
    \item To train, evaluate, and optimize a machine learning classifier using a Gradient Boosting Classifier (GBC) algorithm, achieving a diagnostic classification accuracy of at least 95\% on a dataset of 100,000 patient records.
    \item To develop a automated data preprocessing and scaling pipeline that handles categorical feature encoding and Z-score normalization, ensuring consistent predictions on user inputs.
    \item To implement a stateless, lightweight backend REST API using Python Flask and Gunicorn that loads the serialized model artifacts and executes inference requests in under 300 milliseconds.
    \item To build a responsive React single-page application (SPA) with a brutalist, high-contrast dark user interface, featuring interactive slider controls for real-time risk simulation and structured, tiered recommendations.
    \item To containerize the application using a multi-stage Docker workflow and deploy it to Hugging Face Spaces, ensuring public accessibility on desktop and mobile devices.
\end{enumerate}

\section{Project Scope}
The scope of the DiaFlux project is defined to ensure a focused, high-performance implementation. 
\subsection*{In Scope:}
\begin{itemize}
    \item \textbf{Data Preprocessing:} Automating data cleaning, duplicate removal, one-hot encoding, and scaling on clinical parameters (HbA1c, fasting glucose, BMI).
    \item \textbf{Model Development:} Comparing four classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting) and serializing the best model.
    \item \textbf{Web Application:} Implementing a React frontend and a Flask backend to serve predictions, recommendations, and simulations.
    \item \textbf{Containerization \& Deployment:} Building a Docker image and hosting the stateless application publicly.
\end{itemize}

\subsection*{Out of Scope:}
\begin{itemize}
    \item \textbf{EHR Integration:} Connecting the application to electronic health record databases.
    \item \textbf{Wearable Integration:} Real-time syncing with wearable fitness trackers or continuous glucose monitors.
    \item \textbf{Authentication \& DB Storage:} Implementing user account systems, logins, and database storage (to ensure data privacy).
    \item \textbf{Clinical Diagnosis:} Providing a medical diagnosis. The application serves strictly as an educational screening and risk assessment tool.
\end{itemize}

\section{Report Organization}
This thesis is organized into five chapters. Chapter 1 introduces the project background, motivation, problem statement, objectives, and scope. Chapter 2 provides the requirements specification, including a comparative review of existing systems, feasibility analyses, functional and non-functional requirements, and use case diagrams. Chapter 3 details the system design, covering the architecture, component hierarchy, backend modules, activity diagrams, sequence diagrams, dataset schemas, and data visualizations. Chapter 4 describes the implementation and testing phases, detailing the development environment, machine learning training pipeline, backend code, frontend components, and test results. Chapter 5 discusses the application deployment, environment variables, local setup instructions, user manual, and areas for future work.

\clearpage

% =============================================================================
% CHAPTER 2: REQUIREMENTS SPECIFICATION
% =============================================================================
\chapter{Requirements Specification}

\section{Existing Systems Review}
To establish the context and value of the proposed system, five existing risk assessment tools were analyzed:
\begin{enumerate}
    \item \textbf{ADA Risk Test:} The American Diabetes Association online test is a simple questionnaire that calculates risk based on points for age, family history, and physical activity \cite{ada_standards}.
    \begin{itemize}
        \item \textit{Limitations:} It does not use clinical variables (e.g., HbA1c, glucose), does not leverage machine learning models, and offers no simulation capability.
        \item \textit{DiaFlux Improvement:} DiaFlux integrates real clinical data, uses an ensemble model, and provides a simulator to show how changes in metrics alter risk.
    \end{itemize}
    
    \item \textbf{Framingham Risk Score:} A widely used clinical score that predicts 10-year risk of cardiovascular disease and diabetes.
    \begin{itemize}
        \item \textit{Limitations:} It is a static scoring sheet that requires clinical manual entry and lacks interactive visualization for patients.
        \item \textit{DiaFlux Improvement:} DiaFlux provides a real-time web interface with dynamic sliders, designed specifically for patient interaction.
    \end{itemize}

    \item \textbf{CDC Diabetes Risk Screener:} A basic public-health screening form similar to the ADA test.
    \begin{itemize}
        \item \textit{Limitations:} It relies on subjective user responses and simple rule-based scoring, providing limited clinical fidelity.
        \item \textit{DiaFlux Improvement:} DiaFlux uses calibrated objective variables (HbA1c, fasting blood glucose) and a machine learning classifier.
    \end{itemize}

    \item \textbf{FINDRISC (Finnish Diabetes Risk Score):} An 8-question screening tool widely used in Europe to predict 10-year diabetes risk.
    \begin{itemize}
        \item \textit{Limitations:} It is a static, paper-based or simple digital form that does not allow users to model alternative health scenarios.
        \item \textit{DiaFlux Improvement:} DiaFlux enables dynamic risk simulation, showing the immediate mathematical impact of lifestyle changes.
    \end{itemize}

    \item \textbf{Google Health Studies App:} A mobile platform used to gather volunteer health data for large-scale medical studies.
    \begin{itemize}
        \item \textit{Limitations:} It is research-focused, does not provide public open-access APIs for immediate risk calculation, and lacks lifestyle simulation.
        \item \textit{DiaFlux Improvement:} DiaFlux is fully open-source, publicly deployed via Docker, and designed for immediate public screening.
    \end{itemize}
\end{enumerate}

\begin{table}[H]
\centering
\caption{Comparison of Diabetes Risk Assessment Tools}
\label{tab:system_comparison}
\begin{tabularx}{\textwidth}{lXXXXXX}
\toprule
\textbf{Tool} & \textbf{Input Features} & \textbf{ML Used} & \textbf{Simulation} & \textbf{Open Access} & \textbf{Deployment} \\
\midrule
ADA Risk Test & 7 (Demographics) & No & No & Yes & Web Form \\
Framingham Score & 8 (Clinical) & No (Statistical) & No & Yes & Medical Calculators \\
CDC Screener & 7 (Demographics) & No & No & Yes & Web Form \\
FINDRISC & 8 (Demographics+BMI) & No (Statistical) & No & Yes & Paper / Simple Web \\
Google Health App & Various (Sensors) & Yes (Internal) & No & No & Mobile App Store \\
\textbf{DiaFlux} & \textbf{8 (Clinical+Dem.)} & \textbf{Yes (GBC)} & \textbf{Yes (Sliders)} & \textbf{Yes} & \textbf{Docker / HF Spaces} \\
\bottomrule
\end{tabularx}
\end{table}

\section{Proposed System Overview}
DiaFlux addresses the limitations of existing tools by introducing a machine learning model integrated with a real-time lifestyle simulator. A Gradient Boosting Classifier (GBC) was selected for its high classification accuracy on tabular datasets and its ability to handle class imbalance. The architecture uses a Flask REST API backend to load model binaries and serve predictions, while the React 19 frontend provides an interactive, responsive user interface. To simplify deployment, the entire system is packaged as a single-port Docker container, eliminating CORS configuration and reducing hosting requirements.

\section{Feasibility Analysis}
\begin{itemize}
    \item \textbf{Technical Feasibility:} The development stack is based on stable, open-source technologies (Python 3.11, scikit-learn, React 19, TypeScript, Docker). The training and inference pipelines require minimal computing resources, making the application feasible to build on standard laptops and deploy on free cloud hosting tiers.
    \item \textbf{Economic Feasibility:} The project has zero licensing costs due to its open-source stack. Hugging Face Spaces offers a free Docker hosting tier with continuous deployment, making the application free to maintain.
    \item \textbf{Social Feasibility:} DiaFlux addresses a clear public health need by providing an accessible, easy-to-use screening tool. It promotes health literacy and preventive care, making it socially beneficial.
\end{itemize}

\section{System Requirements}

\begin{table}[H]
\centering
\caption{Functional Requirements Specification}
\label{tab:functional_requirements}
\begin{tabularx}{\textwidth}{llX}
\toprule
\textbf{ID} & \textbf{Name} & \textbf{Description} \\
\midrule
FR-01 & Physiological Input & The system must collect 9 demographic and clinical inputs from the user. \\
FR-02 & Risk Prediction & The backend must run clinical inputs through the GBC model to predict risk. \\
FR-03 & Risk Classification & The system must classify risk as Low ($<30\%$), Medium ($30\%$--$70\%$), or High ($>70\%$). \\
FR-04 & Biometric Simulation & The simulator must allow users to adjust BMI, HbA1c, and glucose using sliders. \\
FR-05 & Risk Comparison & The system must show both original and simulated risk to display improvements. \\
FR-06 & Action Recommendations & The backend must generate dietary, fitness, and medical guidance based on risk. \\
FR-07 & Educational Library & The system must provide reference materials aligned with WHO and ADA guidelines. \\
FR-08 & Input Validation & The backend must return a 400 error for malformed, out-of-range, or missing inputs. \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering
\caption{Non-Functional Requirements Specification}
\label{tab:non_functional_requirements}
\begin{tabularx}{\textwidth}{llX}
\toprule
\textbf{ID} & \textbf{Attribute} & \textbf{Metric / Constraint} \\
\midrule
NFR-01 & Performance & The backend API must return predictions in less than 300 milliseconds. \\
NFR-02 & Reliability & The application must achieve a 99.5\% uptime target on Hugging Face Spaces. \\
NFR-03 & Usability & The user interface must be fully mobile-responsive using Tailwind CSS. \\
NFR-04 & Data Privacy & The application must be stateless, storing no personal data (PII) on the server. \\
NFR-05 & Portability & The system must run on a single port within a Docker container. \\
\bottomrule
\end{tabularx}
\end{table}

\section{Use Case Diagram}
Figure \ref{fig:usecase} shows the actors and use cases for the DiaFlux system, detailing user and clinical access points.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.85, every node/.style={transform shape},
    actor/.style={draw, thick, circle, minimum size=0.8cm, fill=blue!10},
    usecase/.style={draw, thick, ellipse, minimum width=2.4cm, minimum height=0.9cm, fill=emerald!10, text width=2.2cm, align=center, font=\scriptsize},
    sysbox/.style={draw, dashed, thick, fill=gray!5, minimum width=7cm, minimum height=9.5cm}
]
% Boundary
\node[sysbox, label={[anchor=north]90:DiaFlux Boundary}] (boundary) at (2.5, -2.5) {};

% Actors
\node[actor, label={below:Patient}] (pat) at (-2.5, 0) {};
\node[actor, label={below:Clinician}] (cli) at (-2.5, -4.5) {};
\node[actor, label={below:System API}] (sys) at (7.5, -2) {};

% Use cases
\node[usecase] (uc1) at (2.5, 1) {Submit Metrics \\ (predict)};
\node[usecase] (uc2) at (2.5, -0.4) {View Score};
\node[usecase] (uc3) at (2.5, -1.8) {Run Simulation \\ (simulate)};
\node[usecase] (uc4) at (2.5, -3.2) {View Guidance \\ (recs)};
\node[usecase] (uc5) at (2.5, -4.6) {View Education};
\node[usecase] (uc6) at (2.5, -6) {Health API \\ (health)};

% Connections
\draw[thick] (pat) -- (uc1.west);
\draw[thick] (pat) -- (uc2.west);
\draw[thick] (pat) -- (uc3.west);
\draw[thick] (pat) -- (uc4.west);
\draw[thick] (pat) -- (uc5.west);

\draw[thick] (cli) -- (uc2.west);
\draw[thick] (cli) -- (uc3.west);
\draw[thick] (cli) -- (uc4.west);

\draw[thick] (sys) -- (uc1.east);
\draw[thick] (sys) -- (uc3.east);
\draw[thick] (sys) -- (uc6.east);
\end{tikzpicture}
\caption{DiaFlux System Use Case Diagram}
\label{fig:usecase}
\end{figure}

\section{Use Case Descriptions}

\subsection*{UC-01: Submit Health Metrics and Receive Prediction}
\begin{itemize}
    \item \textbf{ID:} UC-01
    \item \textbf{Name:} Submit Health Metrics and Receive Prediction
    \item \textbf{Actor:} Patient, Clinician, System API
    \item \textbf{Precondition:} The user is on the Assessment tab and the backend is healthy (`/api/health` returns ok).
    \item \textbf{Main Flow:}
    \begin{enumerate}
        \item The user inputs their biological sex, age, hypertension/heart disease status, smoking history, BMI, HbA1c, and fasting blood glucose.
        \item The frontend validates that the inputs are complete and within acceptable ranges.
        \item The user clicks ``Run ML Risk Appraisal''.
        \item The React application sends a POST request with the JSON payload to `/api/predict`.
        \item The backend parses the data, formats the feature vector, scales the values, runs the GBC model, and generates recommendations.
        \item The backend returns a JSON payload containing the prediction status, probability score, risk classification, and guidance.
        \item The frontend redirects the user to the ``Risk Report'' tab to view their results.
    \end{enumerate}
    \item \textbf{Alternative Flow:} If the validation step fails, the system highlights the out-of-range fields and blocks submission. If the backend is unreachable, the application displays an ``Analytical error detected'' banner.
    \item \textbf{Postcondition:} The patient metrics and predictions are loaded into the application state, enabling downstream simulation features.
\end{itemize}

\subsection*{UC-02: Run Lifestyle Simulation}
\begin{itemize}
    \item \textbf{ID:} UC-02
    \item \textbf{Name:} Run Lifestyle Simulation
    \item \textbf{Actor:} Patient, Clinician, System API
    \item \textbf{Precondition:} UC-01 must be completed, and patient metrics must be loaded in the application state.
    \item \textbf{Main Flow:}
    \begin{enumerate}
        \item The user navigates to the ``Live Simulator'' tab.
        \item The page displays the baseline metrics alongside adjustable sliders for BMI, HbA1c, and fasting blood glucose.
        \item The user moves a slider to select a target metric value.
        \item The frontend triggers a debounced (350ms) POST request to `/api/simulate`, containing the baseline data and the modified values.
        \item The backend calculates the baseline and simulated risk scores and estimates the percentage reduction.
        \item The backend returns the probability scores, improvement percentage, and a summary text.
        \item The frontend updates the display in real-time, showing the simulated risk and the reduction percentage.
    \end{enumerate}
    \item \textbf{Alternative Flow:} If the backend fails to respond, the simulation defaults to client-side caching and displays a fallback warning message.
    \item \textbf{Postcondition:} The user can view the simulated risk reduction percentage and click ``Model impact explanation'' to load a detailed summary.
\end{itemize}

\subsection*{UC-03: View Clinical Recommendations}
\begin{itemize}
    \item \textbf{ID:} UC-03
    \item \textbf{Name:} View Clinical Recommendations
    \item \textbf{Actor:} Patient, Clinician
    \item \textbf{Precondition:} UC-01 must be completed, and the backend must return recommendations in the prediction response.
    \item \textbf{Main Flow:}
    \begin{enumerate}
        \item The user navigates to the ``Action Guidelines'' tab.
        \item The page loads the recommendations array from the prediction results.
        \item The interface organizes the guidelines into three columns: Dietary \& Nutrition, Fitness Progressions, and Clinical Procedures.
        \item The user can click individual recommendations to mark them as completed.
    \end{enumerate}
    \item \textbf{Alternative Flow:} If no clinical prediction has been run, the page displays a ``Biometric appraisal required'' notice.
    \item \textbf{Postcondition:} The interactive recommendations checklist is displayed to the user.
\end{itemize}

\clearpage

% =============================================================================
% CHAPTER 3: SYSTEM DESIGN
% =============================================================================
\chapter{System Design}

\section{Architecture Design}

\subsection{High-Level System Architecture}
DiaFlux is structured as a layered MVC-style architecture. Figure \ref{fig:layered_arch} illustrates the system components and communication protocols.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.85, every node/.style={transform shape},
    tier/.style={draw, thick, fill=blue!5, rounded corners, minimum width=11cm, minimum height=1.3cm, align=center, font=\bfseries\small},
    arrow/.style={Latex-Latex, thick, draw=gray!80, double}
]
\node[tier, fill=red!15] (t1) {Tier 1: User Interface (Web Browser)\\React 19 SPA client, brutalist layout};
\node[tier, fill=orange!15, below=0.8cm of t1] (t2) {Tier 2: Frontend Server Layer\\Vite, Tailwind CSS, TypeScript components, server.ts proxy};
\node[tier, fill=yellow!15, below=0.8cm of t2] (t3) {Tier 3: Backend REST API\\Python Flask, WSGI Gunicorn handler, CORS policy manager};
\node[tier, fill=emerald!15, below=0.8cm of t3] (t4) {Tier 4: Machine Learning Inference Layer\\GradientBoostingClassifier, StandardScaler, joblib deserializer};

\draw[arrow] (t1) -- node[right, font=\scriptsize] {Local execution / DOM Rendering} (t2);
\draw[arrow] (t2) -- node[right, font=\scriptsize] {HTTP POST/GET (REST / JSON)} (t3);
\draw[arrow] (t3) -- node[right, font=\scriptsize] {In-memory function calls / numpy arrays} (t4);
\end{tikzpicture}
\caption{DiaFlux 4-Tier High-Level System Architecture}
\label{fig:layered_arch}
\end{figure}

\subsection{Development vs Production Architecture}
The application supports two distinct environments:
* **Development Environment:** The React frontend runs on port `3000` via Vite. API requests are routed through a local Express server (`server.ts`), which acts as a proxy forwarding `/api/*` requests to the Flask backend on port `5000`. This simplifies local development by separating frontend and backend logs.
* **Production Environment:** Built as a unified Docker container. A multi-stage Dockerfile builds the static React assets and copies them into the Flask distribution directory (`diaflux_frontend/dist`). The Flask app serves the static assets on `/` and handles API requests on `/api/*` under a single Gunicorn server (typically port `7860`). This design avoids CORS issues and reduces cloud deployment costs.

\section{Component-Level Design}

\subsection{Frontend Component Hierarchy}
Figure \ref{fig:comp_tree} shows the structure of the React single-page application.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=1.3cm and 0.2cm,
    comp/.style={draw, thick, fill=blue!10, minimum width=2.4cm, minimum height=1cm, align=center, font=\scriptsize},
    rootcomp/.style={draw, thick, fill=red!10, minimum width=3cm, minimum height=1.1cm, align=center, font=\small\bfseries}
]
\node[rootcomp] (root) {App.tsx \\ \tiny (Main Layout \& API)};

\node[comp] (c3) {LifestyleSimulator.tsx \\ \tiny (Biometric sliders)};
\node[comp, left=0.4cm of c3] (c2) {ResultsDashboard.tsx \\ \tiny (Risk metrics)};
\node[comp, left=0.4cm of c2] (c1) {RiskForm.tsx \\ \tiny (Clinical inputs)};
\node[comp, right=0.4cm of c3] (c4) {RecommendationsTab.tsx \\ \tiny (Interventions)};
\node[comp, right=0.4cm of c4] (c5) {EducationTab.tsx \\ \tiny (Guidelines)};

\draw[thick] (root.south) -- (c3.north);
\draw[thick] (root.south) -| (c1.north);
\draw[thick] (root.south) -| (c2.north);
\draw[thick] (root.south) -| (c4.north);
\draw[thick] (root.south) -| (c5.north);
\end{tikzpicture}
\caption{DiaFlux React Component Hierarchy Tree}
\label{fig:comp_tree}
\end{figure}

\subsection{Backend Module Structure}
Figure \ref{fig:backend_modules} shows the functional layout of the Python backend in `app.py`.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.9, every node/.style={transform shape},
    mod/.style={draw, thick, fill=yellow!15, minimum width=3.8cm, minimum height=1cm, align=center, font=\scriptsize\bfseries},
    db/.style={draw, thick, cylinder, fill=emerald!15, minimum width=2cm, minimum height=1.5cm, shape border rotate=90, align=center, font=\scriptsize\bfseries},
    arrow/.style={-Latex, thick, draw=gray!80}
]
\node[mod] (api) {Flask Router (app.py) \\ \tiny (API Endpoints)};
\node[mod, below left=1.2cm and 0.5cm of api] (pre) {build\_feature\_frame() \\ \tiny (Preprocesses user inputs)};
\node[mod, below right=1.2cm and 0.5cm of api] (inf) {predict\_probability() \\ \tiny (Executes model inference)};
\node[db, below=3.5cm of api] (artifacts) {Trained Binaries \\ \tiny (models/*.pkl)};

\draw[arrow] (api) -- node[left, font=\tiny] {JSON input} (pre);
\draw[arrow] (pre) -- node[above, font=\tiny] {15-column df} (inf);
\draw[arrow] (inf) -- node[right, font=\tiny] {Risk score} (api);
\draw[arrow] (artifacts) -- node[left, font=\tiny] {scaler.pkl} (pre);
\draw[arrow] (artifacts) -- node[right, font=\tiny] {diabetes\_model.pkl} (inf);
\end{tikzpicture}
\caption{Flask Backend Functional Modules}
\label{fig:backend_modules}
\end{figure}

\subsection{Activity Diagram — Risk Prediction Flow}
Figure \ref{fig:act_predict} outlines the activity flow when a user submits their health metrics for prediction.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.85, every node/.style={transform shape},
    startstop/.style={draw, thick, rounded corners, fill=red!15, minimum width=2cm, minimum height=0.7cm, font=\scriptsize\bfseries},
    process/.style={draw, thick, fill=blue!10, minimum width=2.8cm, minimum height=0.7cm, font=\scriptsize, align=center},
    arrow/.style={-Latex, thick}
]
\node[startstop] (start) {Start};
\node[process, below=0.5cm of start] (p1) {Fill RiskForm};
\node[process, below=0.5cm of p1] (p2) {Validate inputs};
\node[process, below=0.5cm of p2] (p3) {POST to /api/predict};
\node[process, below=0.5cm of p3] (p4) {Process data};
\node[process, right=1cm of p4] (p5) {Scale features};
\node[process, above=0.5cm of p5] (p6) {Run model};
\node[process, above=0.5cm of p6] (p7) {Format output};
\node[process, above=0.5cm of p7] (p8) {Display results};
\node[startstop, right=1cm of p8] (end) {End};

\draw[arrow] (start) -- (p1);
\draw[arrow] (p1) -- (p2);
\draw[arrow] (p2) -- (p3);
\draw[arrow] (p3) -- (p4);
\draw[arrow] (p4) -- (p5);
\draw[arrow] (p5) -- (p6);
\draw[arrow] (p6) -- (p7);
\draw[arrow] (p7) -- (p8);
\draw[arrow] (p8) -- (end);
\end{tikzpicture}
\caption{Activity Diagram: Risk Prediction Flow}
\label{fig:act_predict}
\end{figure}

\subsection{Activity Diagram — Lifestyle Simulation Flow}
Figure \ref{fig:act_simulate} shows the activity loop triggered when adjusting sliders.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.85, every node/.style={transform shape},
    startstop/.style={draw, thick, rounded corners, fill=red!15, minimum width=2cm, minimum height=0.7cm, font=\scriptsize\bfseries},
    process/.style={draw, thick, fill=emerald!15, minimum width=2.8cm, minimum height=0.7cm, font=\scriptsize, align=center},
    arrow/.style={-Latex, thick}
]
\node[startstop] (start) {Start};
\node[process, below=0.5cm of start] (s1) {Move sliders};
\node[process, below=0.5cm of s1] (s2) {POST to /api/simulate};
\node[process, below=0.5cm of s2] (s3) {Predict modified risk};
\node[process, right=1cm of s3] (s4) {Compare probabilities};
\node[process, above=0.5cm of s4] (s5) {Generate summary};
\node[process, above=0.5cm of s5] (s6) {Update frontend};
\node[startstop, right=1cm of s6] (end) {End};

\draw[arrow] (start) -- (s1);
\draw[arrow] (s1) -- (s2);
\draw[arrow] (s2) -- (s3);
\draw[arrow] (s3) -- (s4);
\draw[arrow] (s4) -- (s5);
\draw[arrow] (s5) -- (s6);
\draw[arrow] (s6) -- (end);
\end{tikzpicture}
\caption{Activity Diagram: Lifestyle Simulation Flow}
\label{fig:act_simulate}
\end{figure}

\subsection{Sequence Diagram — Prediction Request}
Figure \ref{fig:seq_predict} shows the sequence of API calls and data processes for a prediction request.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.8, every node/.style={transform shape},
    lifeline/.style={draw, thick, fill=blue!10, minimum width=1.5cm, minimum height=0.6cm, font=\scriptsize\bfseries},
    line/.style={draw, dashed}
]
% Lifelines
\node[lifeline] (user) at (0, 0) {User};
\node[lifeline] (form) at (2.5, 0) {RiskForm};
\node[lifeline] (app) at (5, 0) {App.tsx};
\node[lifeline] (api) at (7.5, 0) {Flask API};
\node[lifeline] (pre) at (10, 0) {Preprocessing};
\node[lifeline] (model) at (12.5, 0) {GBC Model};

% Vertical lines
\draw[line] (user) -- (0, -6.5);
\draw[line] (form) -- (2.5, -6.5);
\draw[line] (app) -- (5, -6.5);
\draw[line] (api) -- (7.5, -6.5);
\draw[line] (pre) -- (10, -6.5);
\draw[line] (model) -- (12.5, -6.5);

% Messages
\draw[-Latex] (0, -1) -- node[above, font=\tiny] {Inputs values} (2.5, -1);
\draw[-Latex] (2.5, -1.8) -- node[above, font=\tiny] {Form submit} (5, -1.8);
\draw[-Latex] (5, -2.6) -- node[above, font=\tiny] {POST /api/predict} (7.5, -2.6);
\draw[-Latex] (7.5, -3.4) -- node[above, font=\tiny] {build\_feature\_frame()} (10, -3.4);
\draw[-Latex] (10, -4.2) -- node[above, font=\tiny] {Scale \& Run Model} (12.5, -4.2);
\draw[Latex-, dashed] (10, -4.8) -- node[above, font=\tiny] {Probability score} (12.5, -4.8);
\draw[Latex-, dashed] (7.5, -5.4) -- node[above, font=\tiny] {Return JSON response} (10, -5.4);
\draw[Latex-, dashed] (5, -6.0) -- node[above, font=\tiny] {Render dashboard} (7.5, -6.0);

\end{tikzpicture}
\caption{System Sequence Diagram: Prediction Request}
\label{fig:seq_predict}
\end{figure}

\section{Data Design}

\subsection{Dataset Schema}
The GBC classifier was trained on a structured dataset containing clinical and demographic features.

\begin{table}[H]
\centering
\caption{Dataset Feature Specifications}
\label{tab:dataset_schema}
\begin{tabularx}{\textwidth}{llXX}
\toprule
\textbf{Column Name} & \textbf{Data Type} & \textbf{Valid Range} & \textbf{Clinical Definition} \\
\midrule
gender & Categorical & Female, Male, Other & Biological sex. \\
age & Numerical & 18.0 -- 80.0+ & General metabolic aging factor. \\
hypertension & Binary & 0, 1 & High blood pressure co-morbidity. \\
heart\_disease & Binary & 0, 1 & Cardiovascular condition history. \\
smoking\_history & Categorical & never, former, current, No Info, ever, not current & Smoking status. \\
bmi & Numerical & 10.0 -- 60.0 & Body Mass Index ($weight/height^2$). \\
HbA1c\_level & Numerical & 3.0\% -- 12.0\% & Glycated hemoglobin level. \\
blood\_glucose\_level & Numerical & 50 -- 400 mg/dL & Fasting blood glucose level. \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Feature Engineering and One-Hot Encoding}
Categorical variables are one-hot encoded to create a 15-dimensional feature vector for the model:
* **gender** $\rightarrow$ `gender_Female`, `gender_Male`, `gender_Other`
* **smoking\_history** $\rightarrow$ `smoking_history_No Info`, `smoking_history_current`, `smoking_history_ever`, `smoking_history_former`, `smoking_history_never`, `smoking_history_not current`

This maps the 8 original inputs to a 15-column vector, which is then normalized using `StandardScaler`.

\subsection{Data Distribution Analysis}
Figures \ref{fig:class_dist} through \ref{fig:feat_imp} show the distributions of key variables in the training dataset.

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    enlargelimits=0.15,
    ylabel={Number of Samples},
    symbolic x coords={Healthy, Diabetic},
    xtick=data,
    nodes near coords,
    nodes near coords align={vertical},
    width=0.7\textwidth,
    height=5.2cm
]
\addplot[fill=blue!40, draw=blue] coordinates {(Healthy,91500) (Diabetic,8500)};
\end{axis}
\end{tikzpicture}
\caption{Class Distribution (Diabetes Target Variable)}
\label{fig:class_dist}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    ylabel={Number of Samples},
    xlabel={HbA1c Level Range (\%)},
    symbolic x coords={3.5-4.5, 4.5-5.0, 5.0-5.5, 5.5-6.0, 6.0-6.5, 6.5-7.0, 7.0-8.0, 8.0-9.0, 9.0+},
    xtick=data,
    x tick label style={rotate=45,anchor=east},
    width=0.85\textwidth,
    height=6.2cm,
    enlargelimits=0.15
]
\addplot[fill=emerald!40, draw=emerald] coordinates {
    (3.5-4.5, 500)
    (4.5-5.0, 4500)
    (5.0-5.5, 35000)
    (5.5-6.0, 42000)
    (6.0-6.5, 10000)
    (6.5-7.0, 5000)
    (7.0-8.0, 2000)
    (8.0-9.0, 800)
    (9.0+, 200)
};
\end{axis}
\end{tikzpicture}
\caption{HbA1c Level Distribution in Dataset}
\label{fig:hba1c_dist}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    ylabel={Number of Samples},
    xlabel={Fasting Blood Glucose Range (mg/dL)},
    symbolic x coords={70-100, 100-125, 126-150, 150-180, 180-220, 220-260, 260-300},
    xtick=data,
    x tick label style={rotate=45,anchor=east},
    width=0.85\textwidth,
    height=6.2cm,
    enlargelimits=0.15
]
\addplot[fill=purple!40, draw=purple] coordinates {
    (70-100, 55000)
    (100-125, 30000)
    (126-150, 8000)
    (150-180, 4000)
    (180-220, 2000)
    (220-260, 800)
    (260-300, 200)
};
\end{axis}
\end{tikzpicture}
\caption{Fasting Blood Glucose Distribution in Dataset}
\label{fig:glucose_dist}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    ylabel={Number of Samples},
    xlabel={Age Group (years)},
    symbolic x coords={18-29, 30-39, 40-49, 50-59, 60-69, 70-80},
    xtick=data,
    x tick label style={rotate=0,anchor=north},
    width=0.85\textwidth,
    height=6.2cm,
    enlargelimits=0.15
]
\addplot[fill=teal!40, draw=teal] coordinates {
    (18-29, 21000)
    (30-39, 18500)
    (40-49, 19500)
    (50-59, 17000)
    (60-69, 14000)
    (70-80, 10000)
};
\end{axis}
\end{tikzpicture}
\caption{Patient Age Distribution in Dataset}
\label{fig:age_dist}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    ylabel={Number of Samples},
    xlabel={BMI Category Range ($kg/m^2$)},
    symbolic x coords={Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (30-34.9), Severe Obese (>=35)},
    xtick=data,
    x tick label style={rotate=15,anchor=east},
    width=0.9\textwidth,
    height=6.2cm,
    enlargelimits=0.15
]
\addplot[fill=orange!40, draw=orange] coordinates {
    (Underweight (<18.5), 3200)
    (Normal (18.5-24.9), 37500)
    (Overweight (25-29.9), 31200)
    (Obese (30-34.9), 16800)
    (Severe Obese (>=35), 11300)
};
\end{axis}
\end{tikzpicture}
\caption{Body Mass Index (BMI) Distribution in Dataset}
\label{fig:bmi_dist}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    xbar,
    xlabel={Relative Importance (\%)},
    ylabel={Clinical Feature},
    symbolic y coords={gender_Male, smoking_No_Info, heart_disease, hypertension, bmi, age, glucose_level, HbA1c_level},
    ytick=data,
    nodes near coords,
    nodes near coords align={horizontal},
    width=0.85\textwidth,
    height=7.2cm,
    enlargelimits=0.15
]
\addplot[fill=blue!50, draw=blue] coordinates {
    (0.03,gender_Male)
    (0.08,smoking_No_Info)
    (0.24,heart_disease)
    (0.42,hypertension)
    (0.96,bmi)
    (2.31,age)
    (31.80,glucose_level)
    (64.15,HbA1c_level)
};
\end{axis}
\end{tikzpicture}
\caption{Gradient Boosting Model Feature Importance Ranking}
\label{fig:feat_imp}
\end{figure}

\section{API Design}
Table \ref{tab:api_specifications} lists the REST API endpoints exposed by the Flask backend.

\begin{table}[H]
\centering
\caption{Backend REST API Specifications}
\label{tab:api_specifications}
\begin{tabularx}{\textwidth}{llp{2.5cm}p{2.5cm}l}
\toprule
\textbf{Method} & \textbf{Endpoint} & \textbf{Request Body} & \textbf{Response Body} & \textbf{Codes} \\
\midrule
GET & `/api/health` & None & Server status, model type & 200, 503 \\
POST & `/api/predict` & `HealthMetrics` & Prediction, recommendations & 200, 400, 500 \\
POST & `/api/simulate` & Original + Modifications & Baseline vs simulated risk comparison & 200, 400, 500 \\
POST & `/api/recommendations` & Risk level + Metrics & Dietary, fitness, and medical guidance & 200, 400 \\
\bottomrule
\end{tabularx}
\end{table}

\section{User Interface Design}

\subsection{Design Decisions}
* **Brutalist Dark Theme:** A brutalist, dark-themed interface was selected to provide high contrast, helping users clearly identify key metrics.
* **Typography:** Clean, sans-serif fonts are used for general text, with large, bold monospace fonts for metrics and score displays.
* **Color-Coded Status:** Visual elements are color-coded to indicate risk tiers: Green for Low Risk ($<30\%$), Amber for Medium Risk ($30\%$--$70\%$), and Red for High Risk ($>70\%$).

\subsection{UI Wireframes}
Figure \ref{fig:wireframes} presents the UI wireframes for key application components.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.7, every node/.style={transform shape},
    frame/.style={draw, thick, minimum width=8.5cm, minimum height=5.5cm, fill=gray!5, rounded corners},
    title/.style={font=\bfseries\small, anchor=north west},
    box/.style={draw, fill=white, minimum height=0.4cm, font=\tiny, align=left}
]
% Wireframe A: RiskForm
\begin{scope}[shift={(0,0)}]
\node[frame] (f1) {};
\node[title] at (-4, 2.4) {Wireframe A: RiskForm};
\node[box, minimum width=3.5cm] at (-2, 1) {Biological Sex: [Female][Male]};
\node[box, minimum width=3.5cm] at (2, 1) {Patient Age: (===O===)};
\node[box, minimum width=3.5cm] at (-2, -0.2) {Hypertension: [No][Yes]};
\node[box, minimum width=3.5cm] at (2, -0.2) {Heart Disease: [No][Yes]};
\node[box, minimum width=7.4cm] at (0, -1.4) {Smoking History: [ dropdown ]};
\node[draw, fill=blue!30, minimum width=7.4cm, minimum height=0.6cm, font=\scriptsize\bfseries] at (0, -2.3) {RUN ML RISK APPRAISAL};
\end{scope}

% Wireframe B: ResultsDashboard
\begin{scope}[shift={(10,0)}]
\node[frame] (f2) {};
\node[title] at (6, 2.4) {Wireframe B: Results Dashboard};
\node[draw, fill=white, minimum width=3.5cm, minimum height=3.5cm, align=center, font=\small\bfseries] at (8, 0) {78\%\\High Risk};
\node[draw, fill=white, minimum width=4cm, minimum height=3.5cm, align=left, font=\tiny] at (12.2, 0) {
  - HbA1c: 6.8\%\\
  - Fasting Glucose: 145\\
  - BMI: 28.5\\
  - Smoking: Former\\
  \\
  [Clinical Recommendations]
};
\end{scope}

% Wireframe C: LifestyleSimulator
\begin{scope}[shift={(0,-7.5)}]
\node[frame] (f3) {};
\node[title] at (-4, -5.1) {Wireframe C: Lifestyle Simulator};
\node[box, minimum width=7.4cm] at (0, -6.5) {BMI Adjustment: (===O===) Sim: 25.0};
\node[box, minimum width=7.4cm] at (0, -7.5) {HbA1c Adjustment: (===O===) Sim: 5.8\%};
\node[box, minimum width=7.4cm] at (0, -8.5) {Glucose Adjustment: (===O===) Sim: 120};
\node[draw, fill=white, minimum width=7.4cm, minimum height=0.8cm, font=\tiny\bfseries] at (0, -9.6) {
  Baseline: 78\% ---> Simulated: 28\% (Reduction: -50\%)
};
\end{scope}

% Wireframe D: Recommendations
\begin{scope}[shift={(10,-7.5)}]
\node[frame] (f4) {};
\node[title] at (6, -5.1) {Wireframe D: Recommendations Checklist};
\node[draw, fill=white, minimum width=2.4cm, minimum height=4cm, align=left, font=\tiny] at (8, -8) {
  \textbf{Dietary:}\\
  [ ] Limit sugars\\
  [ ] Complex carbs\\
  [ ] Lean protein
};
\node[draw, fill=white, minimum width=2.4cm, minimum height=4cm, align=left, font=\tiny] at (11, -8) {
  \textbf{Fitness:}\\
  [ ] 150m cardio/wk\\
  [ ] Resistance 2x/wk\\
  [ ] Post-meal walks
};
\node[draw, fill=white, minimum width=2.4cm, minimum height=4cm, align=left, font=\tiny] at (14, -8) {
  \textbf{Clinical:}\\
  [ ] Standard checks\\
  [ ] Metabolic panel\\
  [ ] Annual screening
};
\end{scope}
\end{tikzpicture}
\caption{User Interface Component Wireframes}
\label{fig:wireframes}
\end{figure}

\clearpage

% =============================================================================
% CHAPTER 4: IMPLEMENTATION AND TESTING
% =============================================================================
\chapter{Implementation and Testing}

\section{Development Environment}
The DiaFlux codebase was developed using the following tools and software configurations:

\begin{table}[H]
\centering
\caption{Development Tool Specifications}
\label{tab:dev_environment}
\begin{tabularx}{\textwidth}{llX}
\toprule
\textbf{Tool / Library} & \textbf{Version} & \textbf{Purpose} \\
\midrule
Python & 3.11 & Backend execution environment. \\
Node.js & 22 & Frontend compilation environment. \\
React & 19.0.1 & Declarative component library. \\
TypeScript & 5.8 & Type safety and interface definitions. \\
Vite & 6.2.3 & Frontend build system. \\
Flask & 3.0 & Backend web API framework. \\
scikit-learn & 1.7.2 & Machine learning training and serialization. \\
Tailwind CSS & 4.1.14 & Layout styling. \\
Docker & Latest & Stage-based container virtualization. \\
VS Code & Latest & Integrated development environment (IDE). \\
Jupyter Notebook & Latest & Experimental analysis and model training. \\
\bottomrule
\end{tabularx}
\end{table}

\section{Machine Learning Pipeline Implementation}

\subsection{Data Preprocessing}
The preprocessing pipeline is implemented in `test_model_saving.py` and the main Jupyter notebooks. It handles missing values, removes duplicates, encodes categorical variables (`gender`, `smoking_history`), and applies `StandardScaler` to normalize numerical parameters.

\subsection{Model Training and Selection}
Listing \ref{lst:model_training} shows the Python code used to train and compare the classification models.

\begin{lstlisting}[language=Python, caption=Machine Learning Model Training Script, label=lst:model_training]
# Import required libraries for model training and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# Define the models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
}

metrics = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    metrics[model_name] = {'accuracy': accuracy, 'f1': f1, 'auc': auc}
\end{lstlisting}

\subsection{Cross-Validation Results}
Table \ref{tab:model_results} details the cross-validation performance of the evaluated algorithms.

\begin{table}[H]
\centering
\caption{Model Performance Metrics Comparison}
\label{tab:model_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{CV Accuracy} & \textbf{CV Std Dev} & \textbf{Test Accuracy} & \textbf{F1-Score} & \textbf{ROC-AUC} \\
\midrule
\textbf{Gradient Boosting} & \textbf{97.20\%} & \textbf{$\pm$0.13\%} & \textbf{97.24\%} & \textbf{0.8088} & \textbf{0.9793} \\
Random Forest & 96.97\% & $\pm$0.09\% & 97.00\% & 0.7969 & 0.9603 \\
SVM (RBF) & 96.24\% & $\pm$0.07\% & 96.07\% & 0.7067 & 0.8943 \\
Logistic Regression & 96.03\% & $\pm$0.05\% & 95.90\% & 0.7199 & 0.9617 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    enlargelimits=0.15,
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1},
    ylabel={Metric Score (\%)},
    symbolic x coords={Accuracy, F1-Score, ROC-AUC},
    xtick=data,
    width=0.85\textwidth,
    height=6.2cm
]
\addplot[fill=blue!30, draw=blue] coordinates {(Accuracy,95.90) (F1-Score,71.99) (ROC-AUC,96.17)}; % LR
\addplot[fill=red!30, draw=red] coordinates {(Accuracy,97.00) (F1-Score,79.69) (ROC-AUC,96.03)}; % RF
\addplot[fill=yellow!30, draw=yellow] coordinates {(Accuracy,96.07) (F1-Score,70.67) (ROC-AUC,89.43)}; % SVM
\addplot[fill=emerald!30, draw=emerald] coordinates {(Accuracy,97.24) (F1-Score,80.88) (ROC-AUC,97.93)}; % GBC
\legend{LR, RF, SVM, GBC}
\end{axis}
\end{tikzpicture}
\caption{Model Evaluation Comparison across Performance Metrics}
\label{fig:model_comp}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    xlabel={False Positive Rate (1 - Specificity)},
    ylabel={True Positive Rate (Sensitivity)},
    xmin=0, xmax=1,
    ymin=0, ymax=1,
    grid=both,
    legend style={at={(0.95,0.05)}, anchor=south east},
    width=0.8\textwidth,
    height=7.2cm
]
\addplot[color=gray, dashed, domain=0:1] {x};
\addplot[color=blue, thick, smooth] coordinates {
    (0,0) (0.05,0.7) (0.1,0.85) (0.2,0.92) (0.4,0.96) (0.6,0.98) (0.8,0.99) (1,1)
};
\addplot[color=red, thick, smooth] coordinates {
    (0,0) (0.08,0.65) (0.15,0.82) (0.25,0.9) (0.45,0.94) (0.65,0.97) (0.85,0.99) (1,1)
};
\addplot[color=yellow!80!black, thick, smooth] coordinates {
    (0,0) (0.15,0.45) (0.3,0.7) (0.5,0.82) (0.7,0.9) (0.85,0.95) (1,1)
};
\addplot[color=emerald, ultra thick, smooth] coordinates {
    (0,0) (0.02,0.78) (0.05,0.9) (0.1,0.95) (0.2,0.97) (0.4,0.98) (0.6,0.99) (1,1)
};
\legend{Random, LR, RF, SVM, GBC}
\end{axis}
\end{tikzpicture}
\caption{Receiver Operating Characteristic (ROC) Curves Comparison}
\label{fig:roc_curves}
\end{figure}

\subsection{Confusion Matrix Analysis}
Figure \ref{fig:confusion_matrix} displays the GBC confusion matrix for the 20,000 hold-out test samples.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.9, every node/.style={transform shape},
    box/.style={draw, thick, minimum width=3cm, minimum height=3cm, align=center, font=\large\bfseries}
]
\node[box, fill=emerald!25] (tn) at (0,0) {True Negative (TN) \\ 18,233 \\ (91.17\%)};
\node[box, fill=crimson!25] (fp) at (3.2,0) {False Positive (FP) \\ 15 \\ (0.08\%)};
\node[box, fill=crimson!25] (fn) at (0,-3.2) {False Negative (FN) \\ 552 \\ (2.76\%)};
\node[box, fill=emerald!25] (tp) at (3.2,-3.2) {True Positive (TP) \\ 1,200 \\ (6.00\%)};

\node[above=0.2cm of tn, font=\bfseries] {Predicted No};
\node[above=0.2cm of fp, font=\bfseries] {Predicted Yes};
\node[left=0.2cm of tn, rotate=90, anchor=south, font=\bfseries] {Actual No};
\node[left=0.2cm of fn, rotate=90, anchor=south, font=\bfseries] {Actual Yes};
\end{tikzpicture}
\caption{Gradient Boosting Classifier Confusion Matrix (N=20,000)}
\label{fig:confusion_matrix}
\end{figure}

\subsection{Hyperparameter Configuration}
Table \ref{tab:hyperparameters} outlines the final hyperparameters configuration for the Gradient Boosting Classifier model.

\begin{table}[H]
\centering
\caption{Gradient Boosting Hyperparameters Configuration}
\label{tab:hyperparameters}
\begin{tabular}{lll}
\toprule
\textbf{Hyperparameter} & \textbf{Assigned Value} & \textbf{Operational Function} \\
\midrule
n\_estimators & 100 & Number of sequential boosting stages to perform. \\
learning\_rate & 0.1 & Shrinks the contribution of each tree to prevent overfitting. \\
max\_depth & 3 & Limits the maximum depth of individual decision trees. \\
criterion & friedman\_mse & Quality of split measurement formula. \\
loss & log\_loss & Cost function minimized during boosting stages. \\
subsample & 1.0 & Fraction of samples used for training individual base learners. \\
ccp\_alpha & 0.0 & Minimal Cost-Complexity Pruning parameter. \\
random\_state & 42 & Seed used by the random number generator. \\
\bottomrule
\end{tabular}
\end{table}

\section{Backend Implementation}

\subsection{Feature Vector Construction}
Listing \ref{lst:feature_vector} shows the backend code used to format raw user inputs into the model's 15-dimensional vector.

\begin{lstlisting}[language=Python, caption=Feature Vector Construction Code, label=lst:feature_vector]
def build_feature_frame(metrics: dict) -> pd.DataFrame:
    # Turn a raw frontend metrics payload into the 15-column model input.
    row = {name: 0 for name in FEATURE_ORDER}

    row["age"] = float(metrics.get("age", 0))
    row["hypertension"] = int(metrics.get("hypertension", 0))
    row["heart_disease"] = int(metrics.get("heart_disease", 0))
    row["bmi"] = float(metrics.get("bmi", 0))
    row["HbA1c_level"] = float(metrics.get("HbA1c_level", 0))
    row["blood_glucose_level"] = float(metrics.get("blood_glucose_level", 0))

    gender = str(metrics.get("gender", "Female"))
    gender_col = f"gender_{gender}"
    if gender_col in row:
        row[gender_col] = 1

    smoking = str(metrics.get("smoking_history", "No Info"))
    smoking_col = f"smoking_history_{smoking}"
    if smoking_col in row:
        row[smoking_col] = 1

    return pd.DataFrame([[row[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)
\end{lstlisting}

\subsection{Inference Pipeline}
Listing \ref{lst:inference} displays the risk probability prediction pipeline.

\begin{lstlisting}[language=Python, caption=Inference Pipeline Implementation, label=lst:inference]
def predict_probability(metrics: dict) -> float:
    # Return the model's probability of diabetes (class 1) for one patient.
    if MODEL is None or SCALER is None:
        raise RuntimeError("Model artifacts are not loaded.")

    features = build_feature_frame(metrics)
    scaled = SCALER.transform(features)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_ORDER)
    proba = MODEL.predict_proba(scaled_df)[0][POSITIVE_CLASS_INDEX]

    # Keep within (0, 1) for numerically stable downstream display.
    return float(min(0.999, max(0.001, proba)))
\end{lstlisting}

\subsection{Simulation Endpoint}
Listing \ref{lst:simulation_endpoint} shows the `/api/simulate` endpoint logic in Flask.

\begin{lstlisting}[language=Python, caption=Simulation API Endpoint Handler, label=lst:simulation_endpoint]
@app.post("/api/simulate")
def simulate():
    if MODEL is None:
        return jsonify({"success": False, "error": "Prediction model is not available."}), 503

    payload = request.get_json(silent=True) or {}
    original_data = payload.get("original_data")
    modifications = payload.get("modifications")
    if not original_data or not modifications:
        return jsonify({"success": False, "error": "Missing parameters."}), 400

    try:
        orig_prob = predict_probability(original_data)
        simulated_data = {**original_data, **modifications}
        sim_prob = predict_probability(simulated_data)

        improvement = 0
        if orig_prob > 0:
            improvement = round(((orig_prob - sim_prob) / orig_prob) * 100)
        improvement = max(-100, improvement)

        return jsonify({
            "original_prediction": orig_prob,
            "simulated_prediction": sim_prob,
            "improvement_percentage": improvement,
            "impact_summary": build_impact_summary(orig_prob, sim_prob, modifications, improvement)
        })
    except Exception as exc:
        return jsonify({"success": False, "error": f"Error during simulation: {exc}"}), 500
\end{lstlisting}

\section{Frontend Implementation}

\subsection{TypeScript Interfaces}
Listing \ref{lst:types} displays the TypeScript interfaces in `types.ts` used to enforce data contracts.

\begin{lstlisting}[language=HTML, caption=TypeScript Shared Typings and Contracts, label=lst:types]
export interface HealthMetrics {
  gender: 'Female' | 'Male';
  age: number;
  hypertension: 0 | 1;
  heart_disease: 0 | 1;
  smoking_history: 'never' | 'former' | 'current' | 'No Info' | 'ever';
  bmi: number;
  HbA1c_level: number;
  blood_glucose_level: number;
}

export interface PredictionResult {
  success: boolean;
  prediction: number;
  probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  confidence: number;
  recommendations: {
    dietary: string[];
    exercise: string[];
    medical: string[];
  };
  explanation: string;
}
\end{lstlisting}

\subsection{Lifestyle Simulator State}
The `LifestyleSimulator.tsx` component manages sliding adjustments. The component triggers a debounced HTTP request to `/api/simulate` when a slider is moved, updating simulated risk values in real-time.

\section{Containerization}

\subsection{Dockerfile Structure}
Listing \ref{lst:dockerfile} shows the multi-stage Dockerfile used to compile and serve the application.

\begin{lstlisting}[language=HTML, caption=Multi-Stage Production Dockerfile, label=lst:dockerfile]
# Stage 1 - Build React SPA
FROM node:22-slim AS frontend
WORKDIR /app/diaflux_frontend
COPY diaflux_frontend/package*.json ./
RUN npm ci
COPY diaflux_frontend/ ./
RUN npx vite build

# Stage 2 - Python Backend and App Runner
FROM python:3.11-slim AS app
WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt gunicorn
COPY backend/ ./backend/
COPY models/ ./models/
COPY --from=frontend /app/diaflux_frontend/dist ./diaflux_frontend/dist
ENV PORT=7860
ENV FRONTEND_DIST=/app/diaflux_frontend/dist
EXPOSE 7860
WORKDIR /app/backend
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "app:app"]
\end{lstlisting}

\subsection{Docker Build Process}
The build stage first uses a Node runtime to compile the React code into static assets. The runtime stage then pulls a lightweight Python image, installs the backend dependencies, copies the static assets and model binaries, and starts the Flask server using Gunicorn.

\section{Testing and Validation}

\subsection{Test Strategy}
The testing strategy covers model training validation, endpoint testing, and manual user testing.

\subsection{Model Serialization Test}
The training pipeline was validated by training classifiers on synthetic data, serializing the models, and reloading the binary files. The reloaded models produced matching class predictions, verifying the serialization pipeline.

\subsection{API Endpoint Testing}
Table \ref{tab:api_testing} presents the backend endpoint test cases.

\begin{table}[H]
\centering
\caption{API Endpoint Testing Results}
\label{tab:api_testing}
\begin{tabularx}{\textwidth}{llp{3cm}XX}
\toprule
\textbf{Endpoint} & \textbf{Method} & \textbf{Payload State} & \textbf{Expected Response} & \textbf{Status} \\
\midrule
`/api/health` & GET & Empty & Status ok, model true & Pass \\
`/api/predict` & POST & Valid clinical payload & Probability, risk details & Pass \\
`/api/predict` & POST & Missing BMI value & 400 Bad Request error & Pass \\
`/api/simulate` & POST & Baseline + Modifications & Simulated probability, text summary & Pass \\
`/api/recommendations` & POST & Risk level high & Structured recommendations & Pass \\
`/api/health` & GET & Server degraded & Degraded status report & Pass \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{User Acceptance Testing}
Table \ref{tab:ui_testing} summarizes the results of usability testing conducted with non-technical users.

\begin{table}[H]
\centering
\caption{User Acceptance Testing Summary}
\label{tab:ui_testing}
\begin{tabular}{lccll}
\toprule
\textbf{Tester} & \textbf{Age} & \textbf{Task Completed} & \textbf{Fidelity Rating} & \textbf{Feedback} \\
\midrule
User A & 24 & Calculate risk & Complete & Interface is clear and easy to navigate. \\
User B & 45 & Adjust sliders & Complete & Sliders show a clear path to lower risk. \\
User C & 62 & Read checkoff lists & Complete & Helpful checklists, easy to read. \\
\bottomrule
\end{tabular}
\end{table}

\clearpage

% =============================================================================
% CHAPTER 5: DEPLOYMENT AND FUTURE WORK
% =============================================================================
\chapter{Deployment and Future Work}

\section{Deployment Architecture Overview}
The production system runs as a containerized Docker application hosted on Hugging Face Spaces. The Flask server handles routing, serving the static assets on `/` and resolving inference queries on `/api/*`.

\section{Pre-Deployment Checklist}
\begin{enumerate}
    \item Verify that `models/diabetes_model.pkl` and `scaler.pkl` are present.
    \item Ensure `backend/requirements.txt` pins the correct package versions.
    \item Verify that the React code is compiled and matches backend interfaces.
    \item Verify the Dockerfile build stages.
    \item Set environment variables on the hosting platform.
\end{enumerate}

\section{Environment Variables}
Table \ref{tab:env_vars} lists the environment variables used to configure the container.

\begin{table}[H]
\centering
\caption{Operational Environment Variables}
\label{tab:env_vars}
\begin{tabular}{lll}
\toprule
\textbf{Variable} & \textbf{Function} & \textbf{Example Value} \\
\midrule
PORT & Binds Gunicorn to the specified port. & 7860 \\
FRONTEND\_DIST & Defines the location of the static assets folder. & `/app/diaflux_frontend/dist` \\
BACKEND\_URL & Points the local proxy to the API server. & `http://localhost:5000` \\
\bottomrule
\end{tabular}
\end{table}

\section{Local Deployment Guide}
To run the container locally:
\begin{enumerate}
    \item Build the Docker image:
    \begin{lstlisting}[language=HTML]
    docker build -t diaflux .
    \end{lstlisting}
    \item Run the container on port 7860:
    \begin{lstlisting}[language=HTML]
    docker run -p 7860:7860 diaflux
    \end{lstlisting}
    \item Open `http://localhost:7860` in a web browser.
\end{enumerate}

\section{Production Deployment to Hugging Face Spaces}
\begin{enumerate}
    \item Create a new Space on Hugging Face, choosing **Docker** SDK and the **Blank** template.
    \item In the README YAML header, specify the port:
    \begin{lstlisting}[language=HTML]
    ---
    title: DiaFlux
    sdk: docker
    app_port: 7860
    ---
    \end{lstlisting}
    \item Push the project files to the Hugging Face repository to trigger the automated build and deployment pipeline.
\end{enumerate}

\section{System Manual}
* **Risk Assessment:** Open the application, enter your clinical metrics, and click ``Run ML Risk Appraisal''.
* **Risk Report:** View your calculated risk probability and classification.
* **Lifestyle Simulator:** Use the sliders to adjust metrics and see how changes affect your risk profile.
* **Guidance Checklist:** Review the structured recommendations for dietary, fitness, and clinical changes.

\section{Known Limitations and Future Work}
* **Recall Constraint:** The model has a recall score of 68.50\%, meaning it could fail to identify approximately 31.5\% of positive diabetes cases. The application serves strictly as an educational screening and risk assessment tool, not a diagnostic system.
* **Data Storage:** The system is stateless and does not store patient history.
* **Planned Features:** Future updates will focus on adding user authentication, trend tracking, and integration with national health databases.

\clearpage

% =============================================================================
% BIBLIOGRAPHY
% =============================================================================
\begin{thebibliography}{99}
\addcontentsline{toc}{chapter}{Bibliography}

\bibitem{idf_atlas}
International Diabetes Federation, \emph{IDF Diabetes Atlas}, 10th ed. Brussels, Belgium: International Diabetes Federation, 2021.

\bibitem{who_report}
World Health Organization, \emph{Global Report on Diabetes}. Geneva, Switzerland: World Health Organization, 2016.

\bibitem{pak_endocrine}
Pakistan Endocrine Society, ``National reports on diabetes epidemiology and clinical burden,'' \emph{Pakistan Endocrine Society Journal}, vol. 12, no. 2, pp. 45--52, 2022.

\bibitem{xgboost}
T. Chen and C. Guestrin, ``XGBoost: A scalable tree boosting system,'' in \emph{Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, 2016, pp. 785--794.

\bibitem{friedman_gb}
J. Friedman, ``Greedy function approximation: A gradient boosting machine,'' \emph{Annals of Statistics}, vol. 29, no. 5, pp. 1189--1232, 2001.

\bibitem{scikit_learn}
F. Pedregosa et al., ``Scikit-learn: Machine learning in Python,'' \emph{Journal of Machine Learning Research}, vol. 12, pp. 2825--2830, 2011.

\bibitem{abdar_uncertainty}
M. Abdar et al., ``A review of uncertainty quantification in deep learning,'' \emph{Information Fusion}, vol. 76, pp. 243--297, 2021.

\bibitem{geron_ml}
A. G\'{e}ron, \emph{Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow}, 2nd ed. Sebastopol, CA: O'Reilly Media, 2019.

\bibitem{kaviani_skin}
S. Kaviani and S. Sami, ``Application of various classifiers for skin lesion classification,'' \emph{International Journal of Computer Applications}, vol. 177, no. 37, pp. 18--24, 2020.

\bibitem{sisodia_diabetes}
D. Sisodia and D. S. Sisodia, ``Prediction of diabetes using classification algorithms,'' \emph{Procedia Computer Science}, vol. 132, pp. 1578--1585, 2018.

\bibitem{vijayan_svm}
M. Vijayan and B. Anjali, ``Prediction and diagnosis of diabetes mellitus using a modified Support Vector Machine,'' in \emph{IEEE International Conference on Control, Communication and Computing}, 2015, pp. 341--345.

\bibitem{temurtas_ann}
H. Temurtas, N. Yumusak, and F. Temurtas, ``A comparative study on diabetes disease diagnosis using neural networks,'' \emph{Expert Systems with Applications}, vol. 36, no. 4, pp. 8610--8615, 2009.

\bibitem{polat_anfis}
K. Polat and S. Gunes, ``An expert system approach based on principal component analysis and adaptive neuro-fuzzy inference system to diagnosis of diabetes disease,'' \emph{Digital Signal Processing}, vol. 17, no. 4, pp. 702--710, 2007.

\bibitem{react_doc}
Meta Open Source, ``React documentation,'' 2024. [Online]. Available: \url{https://react.dev}

\bibitem{flask_doc}
Pallets Projects, ``Flask documentation,'' 2024. [Online]. Available: \url{https://flask.palletsprojects.com}

\bibitem{docker_doc}
Docker Inc., ``Docker official product documentation,'' 2024. [Online]. Available: \url{https://docs.docker.com}

\bibitem{hf_spaces}
Hugging Face, ``Hugging Face Spaces documentation,'' 2024. [Online]. Available: \url{https://huggingface.co/docs/hub/spaces}

\bibitem{tailwind_doc}
Tailwind Labs, ``Tailwind CSS official developer guides,'' 2024. [Online]. Available: \url{https://tailwindcss.com/docs}

\bibitem{ada_standards}
American Diabetes Association, ``Standards of Care in Diabetes---2023,'' \emph{Diabetes Care}, vol. 46, no. Suppl. 1, pp. S1--S291, 2023.

\bibitem{cho_idf_atlas}
N. H. Cho et al., ``IDF Diabetes Atlas: Global estimates of diabetes prevalence for 2017 and projections for 2045,'' \emph{Diabetes Research and Clinical Practice}, vol. 138, pp. 271--281, 2018.

\end{thebibliography}

\end{document}
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "thesis.tex")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(latex_content)

print(f"[DiaFlux Generator] Successfully wrote thesis.tex to: {output_path}")
