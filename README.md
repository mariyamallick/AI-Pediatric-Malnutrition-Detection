# Intelligent Cognitive Alarm Platform

## Team Member
**Mariya Mallick**

## Branch
**MariyaMallick**

---

# Project Overview

The Intelligent Cognitive Alarm Platform is an AI-powered mobile application designed to improve users' wake-up habits. Instead of simply dismissing an alarm, users must complete cognitive challenges such as mathematical problems, logic puzzles, memory challenges, word games, pattern recognition, riddles, and quick quizzes.

The system evaluates user performance, verifies wakefulness, adapts challenge difficulty, analyzes behavior, calculates habit scores, and provides personalized recommendations to encourage healthier wake-up habits.

---

# My Responsibilities (AI/ML)

As the AI/ML developer, my responsibilities included:

- Cognitive Challenge Engine
- Multiple Cognitive Challenge Categories
- Difficulty Management
- Wake-up Verification
- Adaptive Difficulty
- Habit Score Algorithm
- Behavior Analysis
- Recommendation Engine
- AI Service Layer
- Alarm Logic Design
- Alarm Scheduling Design
- AI Workflow Design
- System Architecture
- Database Schema
- AI Testing
- Backend Integration Preparation
- Final AI Module Documentation

---

# Project Progress

## ✅ Week 1 – Project Planning & Architecture

- Designed System Architecture
- Designed Database Schema
- Created HTML Database Schema
- Planned AI Workflow
- Created AI Module Structure
- Initialized AI Project Structure

---

## ✅ Week 2 – AI Challenge & Alarm Design

- Designed Alarm Logic
- Designed Alarm Scheduling Workflow
- Designed Challenge Engine Architecture
- Created Challenge Models
- Added Sample Dataset
- Created AI Testing Module
- Updated AI Documentation

---

## ✅ Week 3 – Cognitive Challenge Engine

Implemented the core Cognitive Challenge Engine.

### Challenge Categories

1. Mathematical Problems
2. Logic Puzzles
3. Memory Challenges
4. Word Games
5. Pattern Recognition
6. Riddles
7. Quick Quizzes

### Features

- Easy difficulty
- Medium difficulty
- Hard difficulty
- Challenge generation
- Answer validation
- Score calculation
- XP reward system
- Category management
- Difficulty management

---

## ✅ Week 4 – Wake-up Verification & Adaptive Difficulty

Implemented the wake-up verification workflow.

### Features

- Multiple challenge verification
- Consecutive correct-answer verification
- Wake-up verification logic
- Adaptive difficulty recommendation
- Challenge flow integration
- Main program integration
- User interaction workflow

The system prevents the alarm from being stopped until the required number of challenges are successfully completed.

---

## ✅ Week 5 – User Intelligence

Developed the user intelligence components.

### Habit Score

- Habit score calculation
- Performance-based scoring
- Wake-up verification consideration
- Response-time consideration
- Streak tracking
- Habit-level classification

### Behavior Analysis

- Attempt tracking
- Correct/incorrect answer tracking
- Response-time tracking
- XP tracking
- Success-rate calculation
- Performance statistics

### Recommendation Engine

- Personalized recommendations
- Performance-based recommendations
- Accuracy-based suggestions
- Response-time-based suggestions
- Habit improvement recommendations

All major intelligence components were integrated into the main AI workflow.

---

## ✅ Week 6 – AI Service Layer

Created a unified AI service layer for integration with the backend.

### Implemented

- `ai_service.py`
- Challenge generation service
- Answer validation service
- Wake-up verification service
- Habit score service
- Behavior analysis service
- Recommendation service
- JSON/API-ready responses
- Independent AI service testing

The AI module was structured so that backend services can communicate with the AI components through a unified interface.

---

## ✅ Week 7 – AI Optimization & Data Handling

Worked on improving the AI workflow and making the system more suitable for continued user interaction.

### Implemented / Worked On

- Performance history handling
- User behavior tracking
- Difficulty recommendation based on performance
- Habit and performance statistics
- AI workflow optimization
- Recommendation improvement
- Testing and debugging of integrated AI components

The AI system was refined to use accumulated performance information when analyzing users and recommending future challenge difficulty.

---

## ✅ Week 8 – Final AI Integration & Documentation

Completed the final AI development and documentation phase.

### Completed

- Finalized AI module structure
- Tested the integrated AI workflow
- Verified challenge generation and validation
- Verified wake-up verification
- Verified habit score calculation
- Verified behavior analysis
- Verified recommendation generation
- Finalized AI service layer
- Prepared AI components for backend integration
- Updated project documentation
- Updated README
- Updated project progress documentation
- Organized AI project files
- Completed testing and debugging of the AI workflow

### Integration Status

The AI module is structured to communicate with the backend through the AI service layer.

Backend and Flutter integration are team-level components and depend on merging the respective backend and frontend implementations.

---

# AI Workflow

```text
                    Alarm Trigger
                         │
                         ▼
                Challenge Generation
                         │
                         ▼
                  User Attempts
                         │
                         ▼
                  Answer Validation
                         │
                         ▼
                Wake-up Verification
                         │
                         ▼
                 Behavior Analysis
                         │
                         ▼
                  Habit Score
                         │
                         ▼
              Adaptive Difficulty
                         │
                         ▼
             Recommendation Engine
                         │
                         ▼
                   AI Service
                         │
                         ▼
                Backend / Flutter