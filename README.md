# 🩸 AnemiaAI — Intelligent Anemia Risk Detection System

> **Final Year Project** — A machine-learning-powered web application for image-based anemia risk assessment using palm/hand, fingernail/nail-bed, and inner-eyelid images.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-success?style=for-the-badge)](https://anemia-detection-cm4p.onrender.com/)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/ClariceDsilva/anemia-detection)

[![Demo Video](https://img.shields.io/badge/YouTube-Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/i7DYuhvqRJw)

---

## 🚀 Live Demo

### 🌐 Try the application

**Live Application:**  
https://anemia-detection-cm4p.onrender.com/

### 🎥 Project Demonstration

**YouTube Demo Video:**  
https://youtu.be/i7DYuhvqRJw

The demonstration video shows the main application workflow, including image upload, validation, anemia risk prediction, and the application's user-facing features.

---

## 📌 Project Overview

AnemiaAI is a web-based machine learning application designed to provide an **image-based preliminary anemia risk assessment**.

The system analyzes visual characteristics from supported biological image regions such as:

- ✋ Palm / hand
- 💅 Fingernail / nail-bed
- 👁️ Inner eyelid / conjunctiva

These regions can exhibit visible changes in coloration and pallor that are relevant to anemia screening.

The application combines:

- Machine Learning
- Computer Vision
- Image preprocessing
- Image validation
- Flask web development
- User authentication
- Prediction history
- Responsive web interfaces

The goal is to provide an accessible educational screening tool while clearly communicating that **anemia cannot be clinically diagnosed from an image alone**.

---

# ✨ Key Features

## 🧠 Machine Learning Prediction

The application uses a trained machine-learning model to classify uploaded images into:

- `Anemic`
- `Normal`

The model also produces an anemia probability and confidence value.

---

## 🖼️ Multiple Image Prediction

Users can upload multiple supported images in a single prediction request.

The system processes the valid images and calculates an overall anemia probability from the individual predictions.

This allows multiple visual regions/images to contribute to a single assessment.

---

## 🛡️ Pre-Prediction Image Validation

Uploaded images are validated **before they are sent to the machine-learning model**.

The validation layer checks for:

1. Corrupt or undecodable images
2. Images that are too small
3. Clearly visible face images
4. Insufficient biological skin/tissue characteristics
5. Excessive edge/detail density associated with documents, screenshots, or unrelated content

If an uploaded batch contains an invalid or irrelevant image, the entire batch is rejected.

This prevents unrelated images from being passed directly to the anemia prediction model.

---

## 👁️ Supported Image Types

The validation system is designed to accept:

- Palm/hand images
- Fingernail/nail-bed images
- Inner-eyelid images

The validator was adjusted to be more tolerant of nail-bed and eyelid images because these images naturally contain less skin-colored area than palm photographs.

---

## 🚫 Irrelevant Image Rejection

The application attempts to prevent users from submitting unrelated images such as:

- Face photographs
- Documents
- Screenshots
- Scenery
- Random/noisy images
- Corrupt image files
- Unsupported file types

Rejected images are not passed to the prediction model.

---

## 👤 User Authentication

The application includes user authentication functionality.

Users can:

- Register/login
- Access authenticated application features
- Logout
- Access their account
- Change their password

Authentication is implemented using Flask-based authentication components.

---

## 📊 Prediction History

Authenticated users can view their previous prediction results.

Prediction history allows users to keep track of previous assessments performed through the application.

---

## 🩺 Symptom-Based Risk Adjustment

The application also provides a symptom assessment component.

User-selected symptom information can be incorporated into the final risk assessment through the application's symptom modifier.

This is used as an additional risk-assessment input and is not intended to replace clinical diagnosis.

---

# 🧬 How the System Works

The overall workflow is:

```text
                ┌──────────────────────┐
                │      User Login      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Upload Images     │
                │ Palm / Nail / Eyelid │
                └──────────┬───────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Image Validation Layer   │
              │                          │
              │ • Decode check           │
              │ • Minimum dimensions     │
              │ • Face detection         │
              │ • Tissue/skin analysis   │
              │ • Edge-density check     │
              └──────────┬───────────────┘
                         │
               ┌─────────┴─────────┐
               │                   │
             Reject               Pass
               │                   │
               ▼                   ▼
        Show validation      ┌───────────────┐
           message           │ Preprocessing │
                             └───────┬───────┘
                                     │
                                     ▼
                             ┌───────────────┐
                             │ ML Prediction │
                             └───────┬───────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ Risk Assessment │
                            └────────┬────────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │ Results + Advice │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Prediction       │
                           │ History          │
                           └──────────────────┘
